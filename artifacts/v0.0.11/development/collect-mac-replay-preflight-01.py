import base64,hashlib,json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
p=json.loads(Path('experiments/v0.0.11/mac-native-replay-protocol.json').read_text())
remote=r"""
import base64,hashlib,json,subprocess
from pathlib import Path
p=json.loads(base64.b64decode(PAYLOAD));root=Path(p['deployment_root']);run=root/'runs/v0.0.11';s=json.loads((run/'h25-supervision-01/supervisor.json').read_text());b=json.loads((run/'h25-preflight-bootstrap.json').read_text())
for where,files in [(root,p['wrapper_files']),(root/'oracle',p['oracle_files'])]:
 for n,h in files.items():assert hashlib.sha256((where/n).read_bytes()).hexdigest()==h,n
for n,d in p['input_files'].items():assert hashlib.sha256((root/'inputs'/n).read_bytes()).hexdigest()==d['sha256'],n
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==b['source_commit']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'oracle',text=True).strip()==p['oracle_commit']
for where in [root,root/'oracle']:assert not subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=where,text=True).strip()
exits=[]
for pid in [b['pid']]+[x['pid'] for x in s['stages']]:
 q=subprocess.run(['ps','-p',str(pid),'-o','pid=,lstart=,command='],capture_output=True,text=True)
 assert q.returncode!=0 or not q.stdout.strip(),q.stdout
 exits.append({'pid':pid,'live':False})
files={}
for f in sorted(run.rglob('*')):
 if f.is_file():
  assert not f.is_symlink();raw=f.read_bytes();files[str(f.relative_to(run))]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
assert sum(x['bytes'] for x in files.values())<p['raw_bytes']
print(json.dumps({'files':files,'process_exits':exits,'source_unchanged':True,'wrapper_files_verified':len(p['wrapper_files']),'oracle_files_verified':len(p['oracle_files']),'input_files_verified':len(p['input_files'])}))
""".replace('PAYLOAD',repr(base64.b64encode(json.dumps(p).encode()).decode()))
r=subprocess.run(['ssh','xbabe4','python3 -'],input=remote,text=True,capture_output=True,timeout=30)
if r.returncode:print(r.stderr);r.check_returncode()
x=json.loads(r.stdout);local=Path('runs/v0.0.11/h25-evidence-01');local.mkdir(exist_ok=False);manifest={}
for n,d in x.pop('files').items():
 raw=base64.b64decode(d.pop('base64'));assert len(raw)==d['bytes'] and hashlib.sha256(raw).hexdigest()==d['sha256'];dest=local/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);manifest[n]=d
out=local/'mac-replay-preflight-01';sup=local/'h25-supervision-01';result=json.loads((out/'result.json').read_text());audit=json.loads((out/'independent-audit.json').read_text());supervisor=json.loads((sup/'supervisor.json').read_text());note=json.loads((out/'post-audit-limitations.json').read_text());bootstrap=json.loads((local/'h25-preflight-bootstrap.json').read_text())
cases=list(ET.parse(sup/'controls.xml').getroot().iter('testcase'));assert len(cases)==21 and all(not list(c) for c in cases)
assert result['physics_steps']==0 and result['physics_calls']==[] and result['status']=='failed_prerequisite'
assert [r['status'] for r in result['rows']]==['failed','not_run','not_run'] and all(r['status']=='not_run' for r in result['native_replay_rows'])
assert supervisor['status']=='stopped_preflight' and supervisor['stages'][2]['returncode']==1 and supervisor['whole_wall_s']<=p['whole_s']
assert audit['status']=='verified_prerequisite_outcome' and audit['new_physics']==0 and supervisor['independent_audit_status']=='passed_saved_outcome'
assert result['versions']==p['versions'] and bootstrap['protocol_sha256']==result['protocol_sha256']
assert all(s['fault'] is None for s in supervisor['stages'])
summary={'status':'verified_failed_prerequisite','source_commit':result['source_commit'],'protocol_sha256':result['protocol_sha256'],'oracle_commit':p['oracle_commit'],'source_controls':{'passed':21,'failed':0,'skipped':0},'preflight':{'attempted':1,'failed':1,'passed':0,'not_run':2},'native_replay':{'attempted':0,'valid':0,'invalid':0,'not_run':6},'physics_steps':0,'rollout_steps':0,'optimizer_updates':0,'corners':result['rows'][0]['corners'],'preflight_worker_wall_s':result['wall_s'],'whole_task_wall_s':supervisor['whole_wall_s'],'setup_before_sync_s':bootstrap['setup_before_sync_s'],'sampled_peak_rss_bytes':max(s['sampled_peak_rss_bytes'] for s in supervisor['stages']),'post_run_audit_wall_s':supervisor['audit_wall_s'],'post_run_audit_limitation':note['audit_guard_limitation'],'raw_files':manifest,'raw_bytes':sum(d['bytes'] for d in manifest.values()),**x}
d=Path('artifacts/v0.0.11/mac-replay-preflight-01');d.mkdir(exist_ok=False)
for name in ['result.json','independent-audit.json','post-audit-limitations.json']:(d/name).write_bytes((out/name).read_bytes())
(d/'controls.xml').write_bytes((sup/'controls.xml').read_bytes());(d/'supervisor.json').write_bytes((sup/'supervisor.json').read_bytes());(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['status','source_controls','preflight','native_replay','physics_steps','corners','whole_task_wall_s','sampled_peak_rss_bytes','process_exits','raw_bytes']}))
