"""Independent saved admission audit; no simulation or source modification."""
import base64,hashlib,json,math,subprocess,sys
from pathlib import Path
p=json.loads(Path('experiments/v0.0.11/resource-admitted-four-context-protocol.json').read_text())
history_path=Path('artifacts/v0.0.11/development/h23-admission-history.json');raw=history_path.read_bytes();history=json.loads(raw)
policy=p['admission_policy'];rows=[]
def evaluate(s,now):
 bad=[]
 assert math.isfinite(s['monotonic_s']) and type(s['affinity_cpus']) is int and s['affinity_cpus']>0
 if not 0<=now-s['monotonic_s']<=policy['fresh_s']:bad.append('stale_or_future')
 if s['load'][0]>policy['load_per_affinity_cpu']*s['affinity_cpus']:bad.append('load')
 for resource,key,limit in [('cpu','some','cpu_some_avg10'),('io','full','io_full_avg10'),('memory','full','memory_full_avg10')]:
  v=s['psi'][resource][key]['avg10'];assert math.isfinite(v) and 0<=v<=100
  if v>policy[limit]:bad.append(resource+'_pressure')
 if s['available_memory_bytes']<policy['minimum_memory_bytes']:bad.append('memory_floor')
 if s['disk_free_bytes']<policy['minimum_disk_bytes']:bad.append('disk_floor')
 if s['jobs']:bad.append('owned_jobs')
 return bad
sha=hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
for n,r in enumerate(history):
 if not r['status'].startswith('deferred'):continue
 assert r['protocol_sha256']==sha
 a=r['admission'];assert a['policy']==policy and len(a['samples'])==2
 s1,s2=a['samples'];spacing=s2['monotonic_s']-s1['monotonic_s']
 bad=['first:'+x for x in evaluate(s1,s1['monotonic_s'])]+['second:'+x for x in evaluate(s2,a['assessed_monotonic_s'])]
 if spacing<policy['spacing_s']:bad.append('spacing')
 assert a['reasons']==bad and a['status']==('deferred' if bad else 'admitted')
 post=r.get('post_checkout')
 if post:
  assert evaluate(post['sample'],post['assessed_monotonic_s'])==post['reasons'] and post['reasons']
 rows.append({'history_index':n,'status':r['status'],'source_commit':r['source_commit'],'initial_pair_status':a['status'],'spacing_s':spacing,'post_reasons':post['reasons'] if post else None,'post_io_full_avg10':post['sample']['psi']['io']['full']['avg10'] if post else None,'post_memory_full_avg10':post['sample']['psi']['memory']['full']['avg10'] if post else None,'observed_transaction_span_s':post['assessed_monotonic_s']-s1['monotonic_s'] if post else a['assessed_monotonic_s']-s1['monotonic_s']})
assert rows
v={'p':p,'source':rows[-1]['source_commit'],'launched':any(r['status']=='launched' for r in history)}
remote=r"""
import base64,hashlib,json,subprocess
from pathlib import Path
v=json.loads(base64.b64decode(PAYLOAD));p=v['p'];root=Path(p['deployment_root'])
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==v['source']
assert subprocess.check_output(['git','rev-parse','--is-shallow-repository'],cwd=root,text=True).strip()=='false'
for n,d in {**p['source_files'],**p['input_files']}.items():
 f=Path(n) if n.startswith('/') else root/n
 assert f.stat().st_size==d['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==d['sha256'],n
for obj,h in p['required_git_objects'].items():assert hashlib.sha256(subprocess.check_output(['git','show',obj],cwd=root)).hexdigest()==h
assert not subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=root,text=True).strip()
if not v['launched']:
 assert not (root/p['output']).exists() and not (root/'runs/v0.0.11/resource-admitted-four-context-deployment.json.gz').exists()
 assert not (root/'experiments/v0.0.11/runtime/.venv').exists()
print(json.dumps({'source_files_verified':len(p['source_files']),'historical_objects_verified':len(p['required_git_objects']),'source_commit':v['source'],'no_formal_outputs':not (root/p['output']).exists()}))
""".replace('PAYLOAD',repr(base64.b64encode(json.dumps(v).encode()).decode()))
r=subprocess.run(['ssh','xbabe1','python3 -'],input=remote,text=True,capture_output=True,timeout=30);r.check_returncode()
a={'status':'verified_saved_deferrals','protocol_sha256':sha,'history_sha256':hashlib.sha256(raw).hexdigest(),'history_bytes':len(raw),'deferrals':rows,'native_worlds':{'attempted':0,'valid':0,'invalid':0,'not_run':4},'controls_not_run':247,'separate_admission_transaction_spans_retained':True,**json.loads(r.stdout)}
out=Path(sys.argv[1]);assert not out.exists(),'Fresh audit label required';out.write_text(json.dumps(a,indent=2)+'\n');print(json.dumps(a))
