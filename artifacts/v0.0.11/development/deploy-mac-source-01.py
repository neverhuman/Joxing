import base64,hashlib,json,subprocess
from pathlib import Path
sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
p=json.loads(Path('experiments/v0.0.11/mac-source-readiness-protocol.json').read_text())
context={'source':sha,'packet':p}
remote=r"""
import base64,hashlib,json,os,platform,re,shutil,subprocess,time,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
v=json.loads(base64.b64decode(PAYLOAD));p=v['packet'];started=time.monotonic()
assert platform.system()=='Darwin' and platform.machine()=='arm64'
repo=json.load(urllib.request.urlopen('https://api.github.com/repos/neverhuman/Joxing',timeout=20));assert repo['private'] is False and repo['visibility']=='public'
files={**p['source_files'],'experiments/v0.0.11/mac-source-readiness-protocol.json':None,'docs/MAC_SOURCE_READINESS_MISSION.md':None,'research/H24.md':None}
def verify(item):
 name,sha=item
 raw=urllib.request.urlopen('https://raw.githubusercontent.com/neverhuman/Joxing/'+v['source']+'/'+name,timeout=20).read()
 actual=hashlib.sha256(raw).hexdigest()
 if sha:assert sha==actual,name
 return {'path':name,'sha256':actual,'bytes':len(raw)}
with ThreadPoolExecutor(max_workers=8) as pool:public=list(pool.map(verify,files.items()))
root=Path(p['deployment_root']);assert root.parent==Path.home()/'joxing'
def capacity():
 raw=subprocess.check_output(['vm_stat'],text=True);page=int(re.search(r'page size of (\d+) bytes',raw).group(1));counts=dict(re.findall(r'^(Pages [^:]+):\s+(\d+)\.',raw,re.M));ram=page*sum(int(counts.get(k,0)) for k in ('Pages free','Pages inactive','Pages speculative'))
 jobs=[line.strip()[:450] for line in subprocess.check_output(['ps','-axo','pid=,stat=,command='],text=True).splitlines() if '/joxing/' in line.lower() and any(k in line for k in ['python','pytest','uv sync'])]
 r={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'load':list(os.getloadavg()),'reclaimable_estimate_bytes':ram,'disk_free_bytes':shutil.disk_usage(Path.home()).free,'joxing_jobs':jobs}
 assert r['load'][0]<=p['maximum_load1'] and ram>=p['minimum_reclaimable_estimate_bytes'] and r['disk_free_bytes']>=p['minimum_disk_bytes'] and not jobs,r
 return r
before=capacity();root.mkdir(parents=True,exist_ok=False)
commands=[['git','init'],['git','remote','add','origin','https://github.com/neverhuman/Joxing.git'],['git','fetch','--depth=1','origin',v['source']],['git','sparse-checkout','set','--no-cone','/src/','/tests/','/scripts/','/pyproject.toml','/uv.lock','/.gitmodules','/experiments/v0.0.11/mac-source-readiness-protocol.json','/third_party/ms-human-700/'],['git','checkout','--detach','FETCH_HEAD'],['git','submodule','update','--init','--depth','1','third_party/ms-human-700']]
for command in commands:
 result=subprocess.run(command,cwd=root,text=True,capture_output=True,timeout=max(1,180-(time.monotonic()-started)))
 assert result.returncode==0,(command,result.stdout,result.stderr)
for name,expected in p['source_files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==v['source']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'third_party/ms-human-700',text=True).strip()==p['model_gitlink']
after=capacity();run=root/'runs/v0.0.11';run.mkdir(parents=True)
command=['python3','scripts/qualify_mac_source.py','--source-sha',v['source']]
with (run/'mac-source-readiness-launch.log').open('x') as log:
 proc=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
report={'schema':'joxing-mac-source-bootstrap-v1','status':'launched','source_commit':v['source'],'protocol_sha256':hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'public_repo_verified':True,'public_files':public,'before':before,'after':after,'checkout_commands':commands,'command':command,'pid':proc.pid,'process_identity':subprocess.check_output(['ps','-p',str(proc.pid),'-o','pid=,lstart=,pgid=,command='],text=True).strip(),'root':str(root),'setup_and_verification_wall_s':time.monotonic()-started,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
(run/'mac-source-bootstrap.json').write_text(json.dumps(report,sort_keys=True,separators=(',',':'))+'\n')
print(json.dumps(report))
""".replace('PAYLOAD',repr(base64.b64encode(json.dumps(context).encode()).decode()))
r=subprocess.run(['ssh','-o','ConnectTimeout=12','xbabe4','python3 -'],input=remote,text=True,capture_output=True,timeout=240)
Path('runs/v0.0.11/mac-source-deploy-output.txt').write_text(r.stdout+r.stderr)
r.check_returncode()
rpt=json.loads(r.stdout)
Path('artifacts/v0.0.11/development/mac-source-bootstrap-01.json').write_text(json.dumps(rpt,sort_keys=True,separators=(',',':'))+'\n')
print(json.dumps({k:rpt[k] for k in ['status','source_commit','pid','root','setup_and_verification_wall_s','after']}))
