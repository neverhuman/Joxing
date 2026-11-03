"""Bounded0physics project-runtime copy to the spare Mac; never starts training."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

repo=Path.cwd();target="/Users/agent/joxing/native-pair-pilot-xbabe4-20261103-01"
out=repo/"runs/v0.0.11/h31-xbabe4-runtime-01";out.mkdir(parents=True,exist_ok=False)
started=time.monotonic();rows=[]
commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
result={"schema":"joxing-xbabe4-readonly-runtime-copy-v1","status":"running","source_commit":commit,"target":target,"physics":0,"stages":rows,"deadline_s":300,"maximum_project_bytes":2*1024**3,"system_packages_changed":False}

def save():
    temp=out/"receipt.json.tmp";temp.write_text(json.dumps(result,indent=2)+"\n");temp.replace(out/"receipt.json")

def run(name,cmd,input=None):
    remaining=300-(time.monotonic()-started)
    if remaining<1:raise TimeoutError("Runtime-copy original300s")
    row={"name":name,"command":cmd,"start_elapsed_s":time.monotonic()-started};rows.append(row);save()
    r=subprocess.run(cmd,input=input,text=True,capture_output=True,timeout=min(120,remaining),check=False)
    row.update(returncode=r.returncode,stdout=r.stdout[-12000:],stderr=r.stderr[-4000:],end_elapsed_s=time.monotonic()-started);save()
    if r.returncode:raise RuntimeError("Owned runtime-copy stage:"+name)
    return r.stdout

try:
    script=f"""
import os,json,subprocess,re
from pathlib import Path
root=Path({target!r});assert not root.exists(),'Fresh project root required'
physical=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True));cpu=os.cpu_count();load=os.getloadavg();pressure=subprocess.check_output(['memory_pressure','-Q'],text=True);free=int(re.search(r'memory free percentage: (\d+)%',pressure)[1])
assert load[0]<=.5*cpu and physical*free//100>=16*1024**3
root.mkdir(parents=True)
print(json.dumps({{'pid':os.getpid(),'cpu':cpu,'load':load,'memory_bytes':physical,'free_percent':free,'root':str(root)}}))
subprocess.run(['git','clone','--depth','1','https://github.com/neverhuman/Joxing.git',str(root/'repo')],check=True,timeout=60)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'repo',text=True).strip()=={commit!r}
subprocess.run(['/Users/agent/.local/share/uv/python/cpython-3.12.3-macos-aarch64-none/bin/python3.12','-m','venv','--without-pip',str(root/'runtime')],check=True,timeout=20)
(root/'inputs').mkdir()
"""
    run('capacity-clone-isolated-venv',['ssh','-o','ConnectTimeout=15','xbabe4','python3 -'],script)
    run('copy-pinned-runtime',['rsync','-a','--exclude','__pycache__','--exclude','__editable__*',str(repo/'.venv/lib/python3.12/site-packages')+'/',f'xbabe4:{target}/runtime/lib/python3.12/site-packages/'])
    source=Path('/Users/bentaylor/joxing/native-pair-motor-pilot-20261103-02/third_party')
    for name in ('ms-human-700','msgym'):
        assert (source/name/'.git').is_dir()
        run('copy-pinned-'+name,['rsync','-a','--exclude','__pycache__',str(source/name)+'/',f'xbabe4:{target}/repo/third_party/{name}/'])
    inputs=repo/'runs/v0.0.6/native-01/inputs'
    for name in ('model.zip','normalizer.pkl'):
        run('copy-input-'+name,['rsync','-a',str(inputs/name),f'xbabe4:{target}/inputs/{name}'])
    script=f"""
import json,platform,importlib.metadata,hashlib,os
from pathlib import Path
root=Path({target!r});paths=[p for p in root.rglob('*') if p.is_file() and not p.is_symlink()];size=sum(p.stat().st_size for p in paths);assert size<=2*1024**3
versions={{'python':platform.python_version(),**{{n:importlib.metadata.version(n) for n in ('numpy','mujoco','torch','stable-baselines3','gymnasium','imageio','imageio-ffmpeg','pillow')}}}}
import torch,mujoco,numpy
assert torch.get_num_threads()==1
inputs={{n:{{'bytes':(root/'inputs'/n).stat().st_size,'sha256':hashlib.sha256((root/'inputs'/n).read_bytes()).hexdigest()}} for n in ('model.zip','normalizer.pkl')}}
print(json.dumps({{'versions':versions,'project_bytes':size,'files':len(paths),'input_files':inputs,'physics':0,'cpu':os.cpu_count(),'load':os.getloadavg()}}))
"""
    remote=f'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 {target}/runtime/bin/python -'
    output=run('import-version-input-cap-preflight',['ssh','-o','ConnectTimeout=15','xbabe4',remote],script)
    result['preflight']=json.loads(output);result['status']='prepared_not_training'
except BaseException as error:
    result.update(status='halted',error=type(error).__name__+': '+str(error))
finally:
    result['wall_s']=time.monotonic()-started;save()
print(json.dumps({k:result.get(k) for k in ('status','error','wall_s','target','physics')}))
if result['status']!='prepared_not_training':sys.exit(1)
