"""Launch the preregistered finite H34reset-placement probe; never touch sealed prior roots."""
import json
import subprocess
from pathlib import Path
repo=Path.cwd();commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
script=r"""
import json,time,subprocess,os,urllib.request
from pathlib import Path
root=Path('/Users/agent/joxing/native-reset-placement-20261103-01/repo');assert not root.parent.exists()
epoch=time.monotonic();commit=COMMIT;original=Path('/Users/agent/joxing/native-pair-pilot-xbabe4-20261103-01/repo')
subprocess.run(['git','clone','--no-hardlinks',str(original),str(root)],check=True,timeout=20,stdout=subprocess.DEVNULL)
subprocess.run(['git','fetch','--depth','1','https://github.com/neverhuman/Joxing.git',commit],cwd=root,check=True,timeout=25,stdout=subprocess.DEVNULL)
subprocess.run(['git','checkout','--detach',commit],cwd=root,check=True,timeout=5,stdout=subprocess.DEVNULL)
for name in ('ms-human-700','msgym'):
 dest=root/'third_party'/name
 subprocess.run(['git','clone','--no-hardlinks',str(original/'third_party'/name),str(dest)],check=True,timeout=15,stdout=subprocess.DEVNULL)
packet='experiments/v0.0.11/native-reset-placement-01.json'
p=json.loads((root/packet).read_text());assert urllib.request.urlopen('https://raw.githubusercontent.com/neverhuman/Joxing/'+commit+'/'+packet,timeout=10).read()==(root/packet).read_bytes()
run=root/'runs/v0.0.11';run.mkdir(parents=True)
protocol_sha=__import__('hashlib').sha256(json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
cmd=[p['runtime_python'],'scripts/qualify_native_reset_placement.py','--packet',packet,'--source-sha',commit,'--protocol-sha256',protocol_sha,'--epoch-start-monotonic',str(epoch)]
env=os.environ|{'PYTHONPATH':str(root/'src')+':'+str(root/'scripts'),'PYTHONDONTWRITEBYTECODE':'1'}
with (run/'reset-launch.log').open('x') as log:
 child=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
start=subprocess.check_output(['ps','-p',str(child.pid),'-o','lstart='],text=True).strip()
result={'status':'launched','source_commit':commit,'host':'xbabe4','root':str(root),'output':str(root/p['output']),'pid':child.pid,'start':start,'original_epoch':epoch,'setup_s':time.monotonic()-epoch,'command':cmd,'new_learning':0,'original_attempt_stays_invalid':True,'protocol_sha256':protocol_sha}
(run/'reset-deployment.json').write_text(json.dumps(result,indent=2)+chr(10));print(json.dumps(result))
""".replace('COMMIT',repr(commit))
__import__('ast').parse(script)
r=subprocess.run(['ssh','-o','ConnectTimeout=15','xbabe4','python3 -'],input=script,text=True,capture_output=True,timeout=65,check=False)
if r.returncode:raise RuntimeError(r.stderr+r.stdout)
v=json.loads(r.stdout.strip().splitlines()[-1]);p=repo/'artifacts/v0.0.11/development/h34-reset-deployment-01.json';assert not p.exists();p.write_text(json.dumps(v,indent=2)+chr(10));print(json.dumps(v))
