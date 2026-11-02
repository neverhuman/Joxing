import base64,hashlib,json,subprocess,time
from pathlib import Path
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
p=json.loads(Path('experiments/v0.0.11/mac-native-replay-protocol.json').read_text())
v={'source':source,'packet':p}
def remote_run(code,timeout=190):
 code=code.replace('PAYLOAD',repr(base64.b64encode(json.dumps(v).encode()).decode()))
 r=subprocess.run(['ssh','-o','ConnectTimeout=12','xbabe4','python3 -'],input=code,text=True,capture_output=True,timeout=timeout)
 if r.returncode:print(r.stdout,r.stderr);r.check_returncode()
 return json.loads(r.stdout)
setup=r"""
import base64,hashlib,json,os,platform,re,shutil,subprocess,time,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
v=json.loads(base64.b64decode(PAYLOAD));p=v['packet'];began=time.monotonic()
assert platform.system()=='Darwin' and platform.machine()=='arm64'
root=Path(p['deployment_root']);assert not root.exists()
def capacity():
 raw=subprocess.check_output(['vm_stat'],text=True);page=int(re.search(r'page size of (\d+) bytes',raw).group(1));counts=dict(re.findall(r'^(Pages [^:]+):\s+(\d+)\.',raw,re.M));ram=page*sum(int(counts.get(k,0)) for k in ('Pages free','Pages inactive','Pages speculative'))
 jobs=[l.strip()[:350] for l in subprocess.check_output(['ps','-axo','pid=,stat=,command='],text=True).splitlines() if '/joxing/' in l.lower() and any(k in l for k in ['python','pytest','uv sync'])]
 r={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'load':list(os.getloadavg()),'reclaimable_estimate_bytes':ram,'disk_free_bytes':shutil.disk_usage(Path.home()).free,'jobs':jobs}
 assert r['load'][0]<=p['maximum_load1'] and ram>=p['minimum_reclaimable_estimate_bytes'] and r['disk_free_bytes']>=p['minimum_disk_bytes'] and not jobs,r
 return r
before=capacity()
def verify(item):
 n,h=item;raw=urllib.request.urlopen('https://raw.githubusercontent.com/neverhuman/Joxing/'+v['source']+'/'+n,timeout=20).read();actual=hashlib.sha256(raw).hexdigest()
 if h:assert actual==h,n
 return {'path':n,'sha256':actual,'bytes':len(raw)}
files={**p['wrapper_files'],'experiments/v0.0.11/mac-native-replay-protocol.json':None,'docs/MAC_NATIVE_REPLAY_MISSION.md':None,'research/H25.md':None}
with ThreadPoolExecutor(max_workers=6) as pool:public=list(pool.map(verify,files.items()))
root.mkdir(parents=True);commands=[]
def run(cmd,cwd=root,stdin=None):
 commands.append({'argv':cmd,'cwd':str(cwd),'stdin':stdin})
 r=subprocess.run(cmd,cwd=cwd,input=stdin,text=True,capture_output=True,timeout=max(1,180-(time.monotonic()-began)));assert r.returncode==0,(cmd,r.stdout,r.stderr)
run(['git','init']);run(['git','remote','add','origin','https://github.com/neverhuman/Joxing.git']);run(['git','fetch','--depth=1','origin',v['source']]);run(['git','sparse-checkout','set','--no-cone','--stdin'],stdin='\n'.join('/'+n for n in files)+'\n/pyproject.toml\n');run(['git','checkout','--detach','FETCH_HEAD'])
oracle=root/'oracle';oracle.mkdir();run(['git','init'],oracle);run(['git','remote','add','origin','https://github.com/neverhuman/Joxing.git'],oracle);run(['git','fetch','--depth=1','origin',p['oracle_commit']],oracle)
names=[n for n in p['oracle_files'] if not n.startswith('third_party/')]+['.gitmodules']
run(['git','sparse-checkout','set','--no-cone','--stdin'],oracle,stdin='\n'.join('/'+n for n in names)+'\n/third_party/ms-human-700/\n/third_party/msgym/\n')
run(['git','checkout','--detach','FETCH_HEAD'],oracle);run(['git','submodule','update','--init','--depth','1','third_party/ms-human-700','third_party/msgym'],oracle)
for n,h in p['oracle_files'].items():assert hashlib.sha256((oracle/n).read_bytes()).hexdigest()==h,n
for n,h in p['wrapper_files'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
run_dir=root/'runs/v0.0.11';run_dir.mkdir(parents=True)
report={'schema':'joxing-h25-preflight-bootstrap-v1','status':'prepared_not_launched','source_commit':v['source'],'protocol_sha256':hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'oracle_commit':p['oracle_commit'],'public_files':public,'oracle_files_verified':len(p['oracle_files']),'commands':commands,'before':before,'after':capacity(),'initial_setup_monotonic_s':began,'setup_wall_so_far_s':time.monotonic()-began,'root':str(root),'native_replay_not_run':6,'physics_steps':0}
(run_dir/'h25-preflight-bootstrap.json').write_text(json.dumps(report,sort_keys=True,separators=(',',':'))+'\n');print(json.dumps(report))
"""
a=remote_run(setup)
Path('runs/v0.0.11/h25-setup-receipt-01.json').write_text(json.dumps(a,indent=2)+'\n')
r=subprocess.run(['scp','-q','-r','runs/v0.0.11/h25-inputs-01','xbabe4:'+p['deployment_root']+'/inputs'],text=True,capture_output=True,timeout=60)
if r.returncode:print(r.stderr);r.check_returncode()
launch=r"""
import base64,hashlib,json,os,re,shutil,subprocess,time
from pathlib import Path
v=json.loads(base64.b64decode(PAYLOAD));p=v['packet'];root=Path(p['deployment_root']);run=root/'runs/v0.0.11';r=json.loads((run/'h25-preflight-bootstrap.json').read_text())
assert r['status']=='prepared_not_launched'
for n,d in p['input_files'].items():
 f=root/'inputs'/n;assert f.stat().st_size==d['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==d['sha256'],n
raw=subprocess.check_output(['vm_stat'],text=True);page=int(re.search(r'page size of (\d+) bytes',raw).group(1));counts=dict(re.findall(r'^(Pages [^:]+):\s+(\d+)\.',raw,re.M));ram=page*sum(int(counts.get(k,0)) for k in ('Pages free','Pages inactive','Pages speculative'))
jobs=[l.strip()[:350] for l in subprocess.check_output(['ps','-axo','pid=,stat=,command='],text=True).splitlines() if '/joxing/' in l.lower() and any(k in l for k in ['python','pytest','uv sync'])]
assert not jobs and ram>=p['minimum_reclaimable_estimate_bytes'] and os.getloadavg()[0]<=p['maximum_load1'] and shutil.disk_usage(root).free>=p['minimum_disk_bytes']
assert time.monotonic()-r['initial_setup_monotonic_s']<180
code=r'''
import json,os,sys,time
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root/'scripts'))
from qualify_mac_source import run_stage
p=json.loads((root/'experiments/v0.0.11/mac-native-replay-protocol.json').read_text());run=root/'runs/v0.0.11';b=json.loads((run/'h25-preflight-bootstrap.json').read_text());out=run/'h25-supervision-01';out.mkdir(exist_ok=False)
env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','PYTHONPATH':str(root),'UV_NO_PROGRESS':'1'}
report={'source_commit':b['source_commit'],'status':'running','stages':[],'native_replay_not_run':6,'physics_steps':0}
def save():(out/'supervisor.json').write_text(json.dumps(report,sort_keys=True,separators=(',',':'))+'\n')
save();started=b['initial_setup_monotonic_s'];oracle=root/'oracle';python=str(oracle/'experiments/v0.0.11/runtime/.venv/bin/python')
stages=[('sync',['uv','sync','--project',str(oracle/'experiments/v0.0.11/runtime'),'--frozen','--python','3.12.3'],max(0,180-(time.monotonic()-started))),('controls',[python,'-m','pytest','-q',str(root/'tests/test_mac_replay.py'),'--junitxml='+str(out/'controls.xml')],p['controls_s']),('preflight',[python,str(root/'scripts/qualify_mac_replay.py'),'--packet',str(root/'experiments/v0.0.11/mac-native-replay-protocol.json'),'--oracle',str(oracle),'--inputs',str(root/'inputs'),'--out',str(root/p['output']),'--source-sha',b['source_commit']],p['stage_s'])]
for name,command,cap in stages:
 if cap<=0:report['status']='setup_deadline';break
 row=run_stage(name,command,out,cap,env,p,started);report['stages'].append(row);save()
 if row['returncode'] or row['fault']:report['status']='stopped_'+name;break
else:report['status']='preflight_complete_native_not_run'
report['whole_wall_s']=time.monotonic()-started;save()
'''
with (run/'h25-launch.log').open('x') as log:proc=subprocess.Popen(['python3','-c',code],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r.update(status='launched',pid=proc.pid,process_identity=subprocess.check_output(['ps','-p',str(proc.pid),'-o','pid=,lstart=,pgid=,command='],text=True).strip()[:500],launch_code=code,prelaunch_load=list(os.getloadavg()),input_files_verified=len(p['input_files']),setup_before_sync_s=time.monotonic()-r['initial_setup_monotonic_s'])
(run/'h25-preflight-bootstrap.json').write_text(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n');print(json.dumps(r))
"""
b=remote_run(launch,timeout=30)
Path('artifacts/v0.0.11/development/mac-replay-preflight-bootstrap-01.json').write_text(json.dumps(b,sort_keys=True,separators=(',',':'))+'\n')
print(json.dumps({k:b[k] for k in ['status','pid','source_commit','setup_before_sync_s','root','prelaunch_load']}))
