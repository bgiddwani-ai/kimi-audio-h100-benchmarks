"""Matched audio QA workload; loopback HTTP timings include server queuing."""
import argparse
import asyncio
import base64
import hashlib
import json
import random
import re
import time
from pathlib import Path

import aiohttp
import numpy as np


def normalized(s):
    return " ".join(re.findall(r"\w+", s.lower()))


def score(pred, gold):
    p, g = normalized(pred), normalized(gold)
    pt, gt = p.split(), g.split()
    from collections import Counter
    overlap = sum((Counter(pt) & Counter(gt)).values())
    f1 = 2 * overlap / (len(pt) + len(gt)) if pt and gt else float(p == g)
    # Binary accuracy requires an unambiguous yes/no answer, not substring matching.
    yn = re.match(r"^(yes|no)\b", p)
    return dict(exact_match=int(p == g), token_f1=f1,
                binary_correct=int(bool(yn) and yn[1] == g) if g in ("yes", "no") else None)


async def run(args):
    root = Path(args.root)
    data_dir = Path(__file__).resolve().parent / "data"
    (root / "results").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (data_dir / "manifest.jsonl").read_text().splitlines()]
    random.Random(20261104).shuffle(rows)
    rows = rows[:args.limit] if args.limit else rows
    payloads = []
    for r in rows:
        raw = (data_dir / Path(r["path"]).name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == r["audio_sha256"]
        question = r["question"] + " Answer briefly."
        content = [
                {"type": "text", "text": question},
                {"type": "input_audio", "input_audio": {
                    "data": base64.b64encode(raw).decode(), "format": "wav"}},
        ]
        payload=dict(model="kimi-audio", temperature=0, max_tokens=64, seed=20261104,
                     stop_token_ids=[151644,151645,151667],stream=True,
                     stream_options={"include_usage":True})
        payload["messages"]=[{"role":"user","content":content}]
        payloads.append(payload)
    out = root / "results" / (args.label + ".jsonl")
    if out.exists():
        raise FileExistsError(f'Refusing to append a second run to {out}')
    connector = aiohttp.TCPConnector(limit=max(64, args.concurrency * 2))
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def request(i, warm=False):
            r, payload = rows[i % len(rows)], payloads[i % len(rows)]
            start = time.perf_counter()
            first, chunks, response_text, usage = None, [], "", {}
            error, status, finish_reason, completed = None, None, None, False
            try:
                endpoint='/v1/chat/completions'
                async with session.post(args.url + endpoint, json=payload) as resp:
                    status = resp.status
                    if status != 200:
                        error = (await resp.text())[:3000]
                    else:
                        async for line in resp.content:
                            if not line.startswith(b"data: "):
                                continue
                            data = line[6:].strip()
                            if data == b"[DONE]":
                                completed = True
                                continue
                            ev = json.loads(data)
                            if ev.get("error"):
                                error = str(ev["error"])
                            if ev.get("usage"):
                                usage = ev["usage"]
                            for c in ev.get("choices", []):
                                if c.get("finish_reason"):
                                    finish_reason = c["finish_reason"]
                                s = c.get("delta", {}).get("content") or c.get("text", "")
                                if s:
                                    now = time.perf_counter()
                                    if first is None:
                                        first = now
                                    chunks.append(now)
                                    response_text += s
            except Exception as exc:
                error = repr(exc)
            end = time.perf_counter()
            return dict(id=r["id"], label=args.label, concurrency=args.concurrency,
                        phase="warmup" if warm else "measured", request_index=i,
                        task="audio_qa",
                        question=r["question"], gold=r["answer"], prediction=response_text,
                        duration_s=r["duration_s"], audio_sha256=r["audio_sha256"],
                        status=status, error=error, finish_reason=finish_reason, completed=completed,
                        success=status == 200 and error is None and first is not None and completed,
                        ttft_ms=(first-start)*1000 if first else None,
                        e2e_ms=(end-start)*1000,
                        chunk_intervals_ms=[(b-a)*1000 for a,b in zip(chunks,chunks[1:])],
                        usage=usage, **score(response_text,r["answer"]))

        # Warm model and transport. Disable MM and prefix caches for cold-input runs.
        warm_rows = await asyncio.gather(*(request(i, True) for i in range(min(2,len(rows)))))
        if not all(r["success"] for r in warm_rows):
            (root / "results" / (args.label + ".warmup.json")).write_text(json.dumps(warm_rows,indent=2))
            raise RuntimeError("Warmup failed: " + str(warm_rows[0].get("error")))
        pending = asyncio.Queue()
        for i in range(len(rows) * args.repeats):
            pending.put_nowait(i)
        measured = []
        async def worker():
            while not pending.empty():
                i = pending.get_nowait()
                r = await request(i)
                measured.append(r)
                with out.open("a") as f:
                    f.write(json.dumps(r) + "\n")
                pending.task_done()
        measured_start_epoch = time.time()
        start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        elapsed = time.perf_counter()-start
        measured_end_epoch = time.time()
    ok = [r for r in measured if r["success"]]
    binary = [r for r in measured if r["binary_correct"] is not None]
    summary = dict(label=args.label, concurrency=args.concurrency, requests=len(measured),
                   successes=len(ok), failures=len(measured)-len(ok), wall_s=elapsed,
                   req_s=len(ok)/elapsed, audio_s_per_s=sum(r["duration_s"] for r in ok)/elapsed,
                   output_tok_s=sum(r["usage"].get("completion_tokens",0) for r in ok)/elapsed,
                   binary_accuracy=np.mean([r["binary_correct"] for r in binary]).item() if binary else None,
                   exact_match=np.mean([r["exact_match"] for r in measured]).item() if measured else None,
                   token_f1=np.mean([r["token_f1"] for r in measured]).item() if measured else None,
                   text_control=False,
                   measured_start_epoch=measured_start_epoch,measured_end_epoch=measured_end_epoch)
    for key in ("ttft_ms","e2e_ms"):
        values=[r[key] for r in ok]
        for q in (50,90,95,99):
            summary[f"{key}_p{q}"] = float(np.percentile(values,q)) if values else None
    (root / "results" / (args.label + ".summary.json")).write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="/work/run")
    p.add_argument("--url",default="http://127.0.0.1:8000")
    p.add_argument("--label",required=True)
    p.add_argument("--concurrency",type=int,default=1)
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--repeats",type=int,default=5)
    asyncio.run(run(p.parse_args()))
