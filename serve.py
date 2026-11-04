"""One entry point for the three measured BF16 configurations."""
import os
import subprocess
import sys
from pathlib import Path

MODEL_ID='moonshotai/Kimi-Audio-7B-Instruct'
REVISION='9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b'
profile=os.environ.get('PROFILE','eager')
if profile not in ('eager','graphs','trt_projector'):
    raise ValueError('PROFILE must be eager, graphs, or trt_projector')
visible=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()
expected=os.environ.get('GPU_UUID')
if len(visible)!=1 or (expected and visible!=[expected]):
    raise RuntimeError(f'Expected one allocated GPU {expected}; visible: {visible}')
driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).splitlines()[0]
compat='/opt/cuda-compat/usr/local/cuda-12.9/compat'
if int(driver.split('.')[0]) in (535,570):
    os.environ['LD_LIBRARY_PATH']=compat+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
from huggingface_hub import snapshot_download
snapshot_download(MODEL_ID,revision=REVISION,local_dir='/work/models/Kimi-Audio-7B-Instruct')
command=[sys.executable,'-m','vllm.entrypoints.cli.main','serve','/work/models/Kimi-Audio-7B-Instruct',
    '--served-model-name','kimi-audio','--trust-remote-code','--dtype','bfloat16',
    '--max-model-len','2048','--max-num-seqs','32','--max-num-batched-tokens','4096',
    '--gpu-memory-utilization','0.80','--mm-processor-cache-gb','0','--no-enable-prefix-caching',
    '--limit-mm-per-prompt','{"audio":1}','--chat-template','/app/chat_template.jinja',
    '--chat-template-content-format','openai','--host',os.environ.get('HOST','0.0.0.0'),
    '--port',os.environ.get('PORT','8000')]
if profile=='eager':
    command.append('--enforce-eager')
elif profile=='trt_projector':
    subprocess.run([sys.executable,'/app/projector.py'],check=True)
    os.environ['KIMI_TRT_PROJECTOR']='/work/engines/projector_fp16_o3.engine'
    command.extend(['--worker-extension-cls','runtime_extensions.TRTWorkerExtension'])
print('PROFILE',profile,'GPU',visible[0],flush=True)
os.execv(sys.executable,command)
