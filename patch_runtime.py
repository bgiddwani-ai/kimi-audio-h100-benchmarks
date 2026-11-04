"""Apply the reviewed tokenizer and variable-length batching compatibility fixes."""
import argparse
import importlib.util
import subprocess
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--patch-dir',type=Path,default=Path(__file__).resolve().parents[1]/'evidence')
args=p.parse_args()
package_root=Path(importlib.util.find_spec('vllm').origin).parent.parent
for name in ('kimi_tokenizer_compatibility.patch','kimi_audio_batching.patch'):
    file=(args.patch_dir/name).resolve()
    command=['patch','--batch','-p0','-i',str(file)]
    check=subprocess.run(command+['--forward','--dry-run'],cwd=package_root,capture_output=True,text=True)
    if check.returncode==0:
        subprocess.run(command+['--forward'],cwd=package_root,check=True)
    else:
        reverse=subprocess.run(command+['--reverse','--dry-run'],cwd=package_root,capture_output=True,text=True)
        if reverse.returncode:
            raise RuntimeError(f'Unexpected vLLM source for {name}: {check.stdout} {check.stderr}')
        print('Already applied:',name)
