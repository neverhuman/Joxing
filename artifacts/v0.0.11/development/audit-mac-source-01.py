import base64,hashlib,json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
p=json.loads(Path('experiments/v0.0.11/mac-source-readiness-protocol.json').read_text())
bootstrap=json.loads(Path('artifacts/v0.0.11/development/mac-source-bootstrap-01.json').read_text())
v={'packet':p,'bootstrap':bootstrap}
remote=r"""
import base64,hashlib,json,subprocess,os
from pathlib import Path
v=json.loads(base64.b64decode(PAYLOAD));p=v['packet'];b=v['bootstrap'];root=Path(p['deployment_root']);out=root/p['output']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==b['source_commit']
assert not subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=root,text=True).strip()
for n,h in p['source_files'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'third_party/ms-human-700',text=True).strip()==p['model_gitlink']
r=json.loads((out/'result.json').read_text());pids=[b['pid']]+[x['pid'] for x in r['stages']]
exits=[]
for pid in pids:
 q=subprocess.run(['ps','-p',str(pid),'-o','pid=,lstart=,command='],text=True,capture_output=True)
 assert q.returncode!=0 or not q.stdout.strip(),q.stdout
 exits.append({'pid':pid,'live':False})
files={}
for path in sorted(out.iterdir()):
 assert path.is_file() and not path.is_symlink()
 raw=path.read_bytes();assert len(raw)<p['raw_bytes'];files[path.name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'base64':base64.b64encode(raw).decode()}
assert sum(x['bytes'] for x in files.values())<p['raw_bytes']
print(json.dumps({'source_files_verified':len(p['source_files']),'model_gitlink':p['model_gitlink'],'process_exits':exits,'raw_files':files,'source_unchanged':True}))
""".replace('PAYLOAD',repr(base64.b64encode(json.dumps(v).encode()).decode()))
x=subprocess.run(['ssh','xbabe4','python3 -'],input=remote,text=True,capture_output=True,timeout=30);x.check_returncode();a=json.loads(x.stdout)
local=Path('runs/v0.0.11/mac-source-evidence-01');local.mkdir(exist_ok=False)
manifest={}
for n,d in a.pop('raw_files').items():
 raw=base64.b64decode(d.pop('base64'));assert len(raw)==d['bytes'] and hashlib.sha256(raw).hexdigest()==d['sha256'];(local/n).write_bytes(raw);manifest[n]=d
r=json.loads((local/'result.json').read_text());proof=json.loads((local/'proof.json').read_text());cases=list(ET.parse(local/'checks.xml').getroot().iter('testcase'))
assert r['status']=='qualified_source_only' and r['source_commit']==bootstrap['source_commit'] and r['source_files']==p['source_files']
assert r['protocol_sha256']==bootstrap['protocol_sha256']==hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert len(cases)==p['test_cases']==r['attempted_test_cases']==proof['collected']==29
assert all(not list(c) for c in cases) and len({(c.attrib['classname'],c.attrib['name']) for c in cases})==29
assert proof['versions']==p['versions'] and proof['physics_calls']==[] and proof['platform']=='Darwin' and proof['machine']=='arm64'
assert all(x['returncode']==0 and x['fault'] is None and x['sampled_peak_rss_bytes']<=p['rss_bytes'] for x in r['stages']) and r['wall_s']<p['whole_s']
report={'status':'verified_source_only','source_commit':r['source_commit'],'protocol_sha256':r['protocol_sha256'],'cases':{'attempted':29,'passed':29,'failed':0,'skipped':0,'not_run':0},'case_names':[c.attrib['classname']+'::'+c.attrib['name'] for c in cases],'versions':proof['versions'],'physics_calls':proof['physics_calls'],'physics_steps':0,'actor_updates':0,'rollouts':0,'supervised_wall_s':r['wall_s'],'setup_verification_wall_s':bootstrap['setup_and_verification_wall_s'],'setup_plus_supervised_s':r['wall_s']+bootstrap['setup_and_verification_wall_s'],'sampled_peak_rss_bytes':max(x['sampled_peak_rss_bytes'] for x in r['stages']),'exact_transfer_files':manifest,'raw_bytes':sum(x['bytes'] for x in manifest.values()),'limits':'Mac source only; no dynamic motor, Linux retirement, learning or games/s claim',**a}
d=Path('artifacts/v0.0.11/mac-source-readiness-01');d.mkdir(exist_ok=False)
(d/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
for name in ['result.json','proof.json','checks.xml']:(d/name).write_bytes((local/name).read_bytes())
print(json.dumps({k:report[k] for k in ['status','source_commit','cases','supervised_wall_s','setup_plus_supervised_s','sampled_peak_rss_bytes','process_exits','raw_bytes']}))
