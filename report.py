"""Validate raw observations and produce the same tables/plots for every run."""
import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

PROFILES=('eager','graphs','trt_projector')
NAMES={'eager':'Eager BF16','graphs':'Compile + CUDA graphs BF16','trt_projector':'TensorRT projector + vLLM BF16'}


def main(root):
    # Original results live directly here; new runs use benchmark.py's results/.
    data=root/'results' if (root/'results').is_dir() else root
    rows=[];raw={};hashes={}
    for profile in PROFILES:
        files=sorted(data.glob(profile+'_c*.summary.json'),key=lambda p:int(p.stem.split('_c')[-1].split('.')[0]))
        if not files:raise RuntimeError(f'No results for {profile}')
        for p in files:
            s=json.loads(p.read_text());name=s['label'];rp=data/(name+'.jsonl')
            records=[json.loads(l) for l in rp.read_text().splitlines()]
            assert len(records)==s['requests'],name
            hashes[name]=hashlib.sha256(rp.read_bytes()).hexdigest();raw[name]=records
            binary=[r['binary_correct'] for r in records if r.get('binary_correct') is not None]
            s.update(profile=profile,binary_accuracy=sum(binary)/len(binary) if binary else None,
                token_f1=sum(r.get('token_f1',0) for r in records)/len(records),
                exact_match=sum(r.get('exact_match',0) for r in records)/len(records))
            for metric in ('ttft_ms','e2e_ms'):
                values=[r[metric] for r in records if r.get('success') and r.get(metric) is not None]
                s[metric+'_max']=max(values) if values else None
                s[metric+'_mean']=sum(values)/len(values) if values else None
            for r in records:
                if r['success']:assert r['completed'] and 0<=r['ttft_ms']<=r['e2e_ms']
            rows.append(s)
    loads=[{r['concurrency'] for r in rows if r['profile']==p} for p in PROFILES]
    assert loads[0]==loads[1]==loads[2],'Profile concurrency matrices differ'
    provenance=data/'provenance.json'
    if provenance.exists():
        original=json.loads(provenance.read_text())
        for cell in original.get('cells',[]):assert hashes[cell['label']]==cell['sha256'],cell['label']
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (data/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (data/'summary.json').write_text(json.dumps(rows,indent=2))
    (data/'raw_sha256.json').write_text(json.dumps(hashes,indent=2))
    plots=data/'plots';plots.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'kimi-report-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.17,'svg.fonttype':'none','savefig.dpi':180})
    colors=['#526a83','#218caa','#9166a7']
    def save(name):
        plt.tight_layout()
        for ext in ['png','svg']:plt.savefig(plots/(name+'.'+ext),bbox_inches='tight')
        plt.close()
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for i,p in enumerate(PROFILES):
        d=sorted([r for r in rows if r['profile']==p],key=lambda r:r['concurrency'])
        for ax,metric in zip(axes,['req_s','ttft_ms_p95']):
            ax.plot([r['concurrency'] for r in d],[r[metric] for r in d],'o-',color=colors[i],label=NAMES[p])
    for ax in axes:
        ticks=sorted(loads[0]);ax.set_xscale('log',base=2);ax.set_xticks(ticks,ticks);ax.set_xlabel('Concurrent requests')
    axes[0].set_ylabel('Requests/s');axes[1].set_ylabel('p95 first-text latency (ms)');axes[0].legend(fontsize=8)
    save('scaling')
    frontier=sorted([a for a in rows if a['successes']==a['requests'] and not any(
        b['successes']==b['requests'] and b['req_s']>=a['req_s'] and b['ttft_ms_p95']<=a['ttft_ms_p95']
        and (b['req_s']>a['req_s'] or b['ttft_ms_p95']<a['ttft_ms_p95']) for b in rows)],key=lambda r:r['ttft_ms_p95'])
    plt.figure(figsize=(9,4.4))
    for i,p in enumerate(PROFILES):
        d=[r for r in rows if r['profile']==p]
        plt.scatter([r['ttft_ms_p95'] for r in d],[r['req_s'] for r in d],color=colors[i],label=NAMES[p])
    plt.plot([r['ttft_ms_p95'] for r in frontier],[r['req_s'] for r in frontier],color='#73828c',label='Observed nondominated frontier')
    plt.xlabel('p95 first-text latency (ms), lower is better');plt.ylabel('Requests/s, higher is better');plt.legend(fontsize=8)
    save('pareto')
    c=8 if 8 in loads[0] else min(loads[0]);fig,axes=plt.subplots(1,2,figsize=(11,4))
    for i,p in enumerate(PROFILES):
        records=raw[f'{p}_c{c}']
        for ax,metric in zip(axes,['ttft_ms','e2e_ms']):
            values=np.sort([r[metric] for r in records if r['success']])
            if len(values):ax.plot(values,np.arange(1,len(values)+1)/len(values),color=colors[i],label=NAMES[p])
    axes[0].set_xlabel('First-text latency (ms)');axes[1].set_xlabel('Complete-answer latency (ms)')
    for ax in axes:ax.set_ylabel('Fraction of completed requests')
    axes[0].legend(fontsize=8);save('latency_cdf')
    title='Original host measurements' if data.name=='host' else 'Container replication measurements'
    lines=[f'# {title}','',f"{len(rows)} cells; {sum(r['requests'] for r in rows):,} attempted requests. "
        '64 unique audio examples, including 32 unique yes/no questions. Each full cell replays them five times; quality sample size remains 32.','',
        '| Configuration | C | OK / attempts | Requests/s | TTFT p50 ms | TTFT p95 ms | E2E p95 ms | Yes/no % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| [{NAMES[r['profile']]}]({r['label']}.jsonl) | {r['concurrency']} | {r['successes']}/{r['requests']} | {r['req_s']:.2f} | {r['ttft_ms_p50']:.2f} | {r['ttft_ms_p95']:.2f} | {r['e2e_ms_p95']:.2f} | {r['binary_accuracy']*100:.3f} |")
    lines+=['','[All metrics CSV](summary.csv) · [JSON](summary.json) · [Raw SHA-256](raw_sha256.json)','',
        'The CSV includes p50/p90/p95/p99/max/mean timing, throughput, output-token rate, exact match and token F1. Raw JSONL retains every answer, reference, usage count and stream-completion status.','',
        '![Concurrency scaling](plots/scaling.png)','','![Latency-throughput frontier](plots/pareto.png)','',
        f'![Latency distributions at concurrency {c}](plots/latency_cdf.png)','',
        'TTFT ends at the first nonempty text delta; E2E ends after the full stream. Both include queueing and audio preprocessing. Inputs are preloaded/base64-encoded before timing. Greedy decoding, max 64 output tokens, stop IDs 151644/151645/151667, two excluded warmups per cell. Caches are disabled. Natural answer lengths vary.','',
        'TensorRT replaces the acoustic projector with an FP16-enabled plan; the vLLM encoder and decoder remain BF16. This port differs from upstream dual-stream Kimi-Audio. These short held-out trials do not establish a production SLA or general quality equivalence.','']
    (data/'README.md').write_text('\n'.join(lines))
    print(json.dumps({'cells':len(rows),'requests':sum(r['requests'] for r in rows),'successful':sum(r['successes'] for r in rows),'report':str(data/'README.md')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
