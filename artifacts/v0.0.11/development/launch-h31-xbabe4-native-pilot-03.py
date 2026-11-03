"""Public-source verified once-only launch on the isolated larger Mac."""
import json
import subprocess
from pathlib import Path

repo=Path.cwd();commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
script=r"""
import hashlib,json,os,subprocess,time,urllib.request
from pathlib import Path
base=Path('/Users/agent/joxing/native-pair-pilot-xbabe4-20261103-01');root=base/'repo'
epoch=time.monotonic()
commit=COMMIT
subprocess.run(['git','fetch','--depth','1','origin',commit],cwd=root,check=True,timeout=30,stdout=subprocess.DEVNULL)
subprocess.run(['git','checkout','--detach',commit],cwd=root,check=True,timeout=10,stdout=subprocess.DEVNULL)
packet='experiments/v0.0.11/native-pair-motor-pilot-03.json'
p=json.loads((root/packet).read_text());out=root/p['output'];assert not out.exists(),'Fresh formal native output required'
blob=urllib.request.urlopen('https://raw.githubusercontent.com/neverhuman/Joxing/'+commit+'/'+packet,timeout=15).read();assert blob==(root/packet).read_bytes()
site=base/'runtime/lib/python3.12/site-packages'
files=[f for f in sorted(site.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and not f.name.startswith('__editable__')]
manifest={str(f.relative_to(site)):{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files}
sha=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert sha==p['runtime_bundle_provenance']['source_manifest_sha256'] and len(files)==22554
for f in files:os.chmod(f,f.stat().st_mode & ~0o222)
for f in reversed(sorted([f for f in site.rglob('*') if f.is_dir()])):os.chmod(f,f.stat().st_mode & ~0o222)
os.chmod(site,site.stat().st_mode & ~0o222)
for f in (base/'inputs').iterdir():os.chmod(f,f.stat().st_mode & ~0o222)
for name,r in {**p['source_files'],**p['input_files'],**p['runtime_file_hashes']}.items():
 f=Path(name) if name.startswith('/') else root/name
 assert f.stat().st_size==r['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'],name
protocol=hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
run=root/'runs/v0.0.11';run.mkdir(parents=True,exist_ok=True)
cmd=[p['runtime_python'],'scripts/qualify_native_pair_pilot.py','--packet',packet,'--source-sha',commit,'--protocol-sha256',protocol,'--epoch-start-monotonic',str(epoch)]
env=os.environ|{'PYTHONPATH':str(root/'src')+':'+str(root/'scripts'),'PYTHONDONTWRITEBYTECODE':'1'}
with (run/'native-pair-pilot03-launch.log').open('x') as log:
 child=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
start=subprocess.check_output(['ps','-p',str(child.pid),'-o','lstart='],text=True).strip()
value={'schema':p['schema'],'status':'launched','host':'xbabe4','root':str(root),'output':str(out),'source_commit':commit,'protocol_sha256':protocol,'runtime_manifest_sha256':sha,'runtime_regular_files':len(files),'source_files_verified':len(p['source_files']),'pid':child.pid,'start':start,'command':cmd,'original_epoch':epoch,'setup_s':time.monotonic()-epoch,'physics':'not_yet_observed'}
(run/'native-pair-pilot03-deployment.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))
""".replace('COMMIT',repr(commit))
r=subprocess.run(['ssh','-o','ConnectTimeout=15','xbabe4','python3 -'],input=script,text=True,capture_output=True,timeout=75,check=False)
path=repo/'artifacts/v0.0.11/development/h31-xbabe4-native-pilot03-deployment.json';assert not path.exists()
if r.returncode:
 path.write_text(json.dumps({'status':'bootstrap_failed','returncode':r.returncode,'stderr':r.stderr,'stdout':r.stdout,'source_commit':commit},indent=2)+'\n');raise RuntimeError(r.stderr)
value=json.loads(r.stdout.strip().splitlines()[-1]);path.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
