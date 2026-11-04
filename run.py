"""Run all three profiles with Docker or Kubernetes; host needs only Python 3."""
import argparse
import datetime
import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROFILES=('eager','graphs','trt_projector')
DEFAULT_GPU='GPU-fbf3e08a-e987-9a86-7d29-5c6b750872e2'


def command(args,**kwargs):
    return subprocess.run([str(a) for a in args],check=True,text=True,**kwargs)


def main(args):
    run_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+str(os.getpid())
    image=os.environ.get('IMAGE','kimi-audio-bench:20261104')
    gpu=os.environ.get('GPU_UUID',DEFAULT_GPU)
    work=Path(args.work_dir).resolve();work.mkdir(parents=True,exist_ok=True)
    (work/'cache').mkdir(exist_ok=True)
    output=work/'runs'/run_id;output.mkdir(parents=True)
    remote='/work/runs/'+run_id
    meta={'run_id':run_id,'runtime':args.runtime,'image':image,'gpu_uuid':gpu,'profiles':list(PROFILES),
        'concurrencies':args.concurrencies,'requests_per_cell':args.limit*args.repeats,
        'benchmark_client':'Executed inside the serving container/pod; loopback HTTP',
        'source_revision':(subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip() if (ROOT/'.git').exists() else 'uncommitted'),
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [*ROOT.glob('*.py'),ROOT/'Dockerfile',ROOT/'k8s.yaml']}}
    cs=[int(x) for x in args.concurrencies.split(',')]
    if not cs or min(cs)<1 or len(set(cs))!=len(cs):raise ValueError('Use distinct positive concurrency levels')
    if not 1<=args.limit<=64 or args.repeats<1:raise ValueError('Use 1-64 examples and at least one replay')
    if args.runtime=='docker':
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1',8033))==0:raise RuntimeError('Port 8033 is occupied; stop that service before running this benchmark')
        meta['image_id']=command(['docker','image','inspect','--format','{{.Id}}',image],capture_output=True).stdout.strip()
    else:
        # The exact-UUID resource claim must be backed by the NVIDIA DRA driver.
        command(['kubectl','get','deviceclass','gpu.nvidia.com'],stdout=subprocess.DEVNULL)
        manifest=(ROOT/'k8s.yaml').read_text().replace('IMAGE_PLACEHOLDER',image).replace('GPU_UUID_PLACEHOLDER',gpu)
    owned=None
    try:
        if args.runtime=='k8s':command(['kubectl','apply','-f','-'],input=manifest)
        for profile in PROFILES:
            print('RUN',profile,flush=True)
            if args.runtime=='docker':
                owned='kimi-bench-'+run_id+'-'+profile
                command(['docker','run','-d','--name',owned,'--gpus','device='+gpu,'--ipc=host','--network','host',
                    '--mount',f'type=bind,src={work},dst=/work',
                    '--mount',f'type=bind,src={work / "cache"},dst=/root/.cache',
                    '-e','PROFILE='+profile,'-e','GPU_UUID='+gpu,
                    '-e','HOST=127.0.0.1','-e','PORT=8033',image],stdout=subprocess.DEVNULL)
                prefix=['docker','exec',owned];url='http://127.0.0.1:8033'
                started=time.monotonic()
                for _ in range(900):
                    if command(['docker','inspect','--format','{{.State.Running}}',owned],capture_output=True).stdout.strip()!='true':
                        command(['docker','logs',owned]);raise RuntimeError('Server exited during startup')
                    try:
                        with urllib.request.urlopen(url+'/health',timeout=2) as resp:
                            if resp.status==200:break
                    except Exception:pass
                    time.sleep(1)
                else:raise TimeoutError('Server did not become ready within 900 seconds')
                meta.setdefault('startup_s',{})[profile]=time.monotonic()-started
            else:
                command(['kubectl','set','env','deployment/kimi-audio','PROFILE='+profile])
                command(['kubectl','rollout','status','deployment/kimi-audio','--timeout=900s'])
                prefix=['kubectl','exec','deployment/kimi-audio','--'];url='http://127.0.0.1:8000'
                command(prefix+['mkdir','-p',remote])
            for c in cs:
                command(prefix+['python','/app/benchmark.py','--root',remote,'--url',url,
                    '--label',f'{profile}_c{c}','--concurrency',str(c),'--limit',str(args.limit),'--repeats',str(args.repeats)])
            if profile==PROFILES[-1]:
                command(prefix+['python','/app/report.py','--root',remote])
                packages=command(prefix+['python','-c',"import importlib.metadata as m,json,tensorrt;print(json.dumps({**{k:m.version(k) for k in ['vllm','torch','transformers','torchcodec']},'tensorrt':tensorrt.__version__}))"],capture_output=True)
                meta['runtime_versions']=json.loads(packages.stdout)
                meta['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
                # Write within the container: bind-mounted results can be root-owned.
                writer="from pathlib import Path;import sys;d=Path(sys.argv[1])/'results';(d/'provenance.json').write_text(sys.argv[2]);p=Path('/work/engines/projector_fp16_o3.json');(d/'projector_build.json').write_bytes(p.read_bytes()) if p.exists() else None"
                command(prefix+['python','-c',writer,remote,json.dumps(meta,indent=2)])
            if args.runtime=='docker':
                logs=command(['docker','logs',owned],capture_output=True)
                (output/(profile+'.server.log')).write_text(logs.stdout+logs.stderr)
                command(['docker','stop','-t','45',owned],stdout=subprocess.DEVNULL)
                command(['docker','rm',owned],stdout=subprocess.DEVNULL);owned=None
        if args.runtime=='k8s':
            pod=command(['kubectl','get','pod','-l','app=kimi-audio','-o','jsonpath={.items[0].metadata.name}'],capture_output=True).stdout
            command(['kubectl','cp',pod+':'+remote+'/.',str(output)])
        print('RESULTS',output/'results',flush=True)
    finally:
        if owned:
            subprocess.run(['docker','stop','-t','45',owned],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            subprocess.run(['docker','rm',owned],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if args.runtime=='k8s':
            # Keep the model/results PVC, release the GPU after the benchmark.
            subprocess.run(['kubectl','scale','deployment/kimi-audio','--replicas=0'],check=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('runtime',choices=['docker','k8s'])
    p.add_argument('--work-dir',default='.work')
    p.add_argument('--concurrencies',default='1,2,4,8,16,32,64')
    p.add_argument('--limit',type=int,default=64)
    p.add_argument('--repeats',type=int,default=5)
    main(p.parse_args())
