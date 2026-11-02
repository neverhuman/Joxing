import base64
import hashlib
import json
import subprocess
from pathlib import Path

packet = json.loads(Path('experiments/v0.0.11/four-native-context-protocol.json').read_text())
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
names = [n for n in packet['source_files'] if not n.startswith('third_party/')] + ['experiments/v0.0.11/four-native-context-protocol.json', 'docs/FOUR_NATIVE_CONTEXT_MISSION.md', 'research/README.md', 'research/H22.md', 'docs/NATIVE_CONTEXT_TRANSFER_RESULTS.md']
payload = {'protocol': packet, 'commit': commit, 'public': {n: {'bytes': Path(n).stat().st_size, 'sha256': hashlib.sha256(Path(n).read_bytes()).hexdigest()} for n in names}}
remote = r'''
import base64,json,hashlib,subprocess,os,time,shutil,urllib.request
from pathlib import Path
p=json.loads(base64.b64decode(PAYLOAD));protocol=p['protocol']; root=Path(protocol['deployment_root'])
from concurrent.futures import ThreadPoolExecutor
def verify_public(item):
 name,d=item
 raw=urllib.request.urlopen('https://raw.githubusercontent.com/neverhuman/Joxing/'+p['commit']+'/'+name,timeout=25).read()
 assert len(raw)==d['bytes'] and hashlib.sha256(raw).hexdigest()==d['sha256'],name
 return {'path':name,**d,'anonymous':True}
with ThreadPoolExecutor(max_workers=8) as pool:public=list(pool.map(verify_public,p['public'].items()))
jobs=[]
for x in Path('/proc').iterdir():
 if not x.name.isdigit():continue
 try:
  cwd=str((x/'cwd').resolve())
  cmd=(x/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
  if ('/home/ubuntu/joxing/' in cmd or cwd.startswith('/home/ubuntu/joxing/')) and any(k in cmd for k in ('python','uv run','timeout')): jobs.append({'pid':int(x.name),'command':cmd[:350]})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not jobs, jobs
available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
assert available>=protocol['minimum_memory_bytes']
assert shutil.disk_usage(root.parent).free>=protocol['minimum_disk_bytes']
root.mkdir(exist_ok=False)
commands=[['git','init',str(root)],['git','remote','add','origin','https://github.com/neverhuman/Joxing.git'],['git','fetch','origin',p['commit']],['git','checkout','--detach','FETCH_HEAD'],['git','submodule','update','--init','--depth','1','third_party/ms-human-700','third_party/msgym']]
for i,cmd in enumerate(commands):
 r=subprocess.run(cmd,cwd=None if i==0 else root,text=True,capture_output=True,timeout=180)
 if r.returncode:raise RuntimeError(r.stdout+r.stderr)
assert subprocess.check_output(['git','rev-parse','--is-shallow-repository'],cwd=root,text=True).strip()=='false'
for obj,expected in protocol['required_git_objects'].items():
 assert hashlib.sha256(subprocess.check_output(['git','show',obj],cwd=root)).hexdigest()==expected
checks=[]
for name,d in {**protocol['source_files'],**protocol['input_files']}.items():
 f=Path(name) if name.startswith('/') else root/name
 assert f.stat().st_size==d['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==d['sha256'],name
 checks.append({'path':name,**d})
sha=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
cmd=['python3','scripts/qualify_deferred_trial.py','--packet','experiments/v0.0.11/four-native-context-protocol.json','--protocol-sha256',sha,'--source-sha',p['commit']]
run=root/'runs/v0.0.11';run.mkdir(parents=True,exist_ok=True)
with (run/'four-native-context-launch.log').open('w') as log:
 proc=subprocess.Popen(cmd,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()
report={'schema':'joxing-four-native-context-deployment-v1','status':'launched','source_commit':p['commit'],'protocol_sha256':sha,'root':str(root),'public_files':public,'verified_files':checks,'process':{'pid':proc.pid,'start_ticks':int(st[19])},'command':cmd,'checkout_commands':commands,'load':os.getloadavg(),'available_memory_bytes':available,'free_disk_bytes':shutil.disk_usage(root).free,'other_owned_jobs':jobs,'authorized_preparation_steps':800,'new_rollout_steps':0,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
import sys
sys.path.insert(0,str(root/'scripts'))
from receipt_encoding import MODE,encode_bootstrap,compact_bytes
assert protocol['receipt_wire_encoding']==MODE
blob,metadata=encode_bootstrap(report)
wire=run/'four-native-context-deployment.json.gz'
temporary=wire.with_suffix(wire.suffix+'.tmp');temporary.write_bytes(blob);temporary.replace(wire)
(run/'four-native-context-deployment.encoding.json').write_bytes(compact_bytes(metadata))
print(json.dumps(report))
'''
remote=remote.replace('PAYLOAD',repr(base64.b64encode(json.dumps(payload).encode()).decode()))
r=subprocess.run(['ssh','-o','ConnectTimeout=15','xbabe1','python3 -'],input=remote,text=True,capture_output=True,timeout=420)
Path('runs/v0.0.11/four-native-context-deploy-output.txt').write_text(r.stdout+r.stderr)
print(r.stderr);r.check_returncode()
report=json.loads(r.stdout)
Path('artifacts/v0.0.11/development/four-native-context-deployment-01.json').write_text(json.dumps(report,indent=2)+'\n')

print(json.dumps({k:report[k] for k in ['status','source_commit','protocol_sha256','root','process','load','available_memory_bytes','free_disk_bytes','other_owned_jobs','utc']}))
