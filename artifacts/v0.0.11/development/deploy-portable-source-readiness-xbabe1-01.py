import base64
import hashlib
import json
import subprocess
from pathlib import Path

packet = json.loads(
    Path(
        "experiments/v0.0.11/portable-linux-source-readiness-xbabe1-protocol.json"
    ).read_text()
)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
names = [n for n in packet["source_files"] if not n.startswith("third_party/")] + [
    "experiments/v0.0.11/portable-linux-source-readiness-xbabe1-protocol.json",
    "docs/PORTABLE_LINUX_SOURCE_READINESS_MISSION.md",
    "research/README.md",
    "research/H29.md",
    "docs/RESOURCE_ADMISSION_RESULTS.md",
]
history_path = Path("artifacts/v0.0.11/development/h29-xbabe1-admission-history.json")
prior = json.loads(history_path.read_text()) if history_path.exists() else []
assert len(json.dumps(prior).encode()) <= 12288, (
    "Admission history reserve exhausted; no new check"
)
payload = {
    "prior_admission_receipts": prior,
    "protocol": packet,
    "commit": commit,
    "public": {
        n: {
            "bytes": Path(n).stat().st_size,
            "sha256": hashlib.sha256(Path(n).read_bytes()).hexdigest(),
        }
        for n in names
    },
}
remote = r"""
import base64,json,hashlib,subprocess,os,time,shutil,urllib.request,resource,signal
from pathlib import Path
p=json.loads(base64.b64decode(PAYLOAD));protocol=p['protocol']; root=Path(protocol['deployment_root']); epoch=time.monotonic()
from concurrent.futures import ThreadPoolExecutor
signal.alarm(60)  # Whole bootstrap/setup cap, never reset by fetch/clone stages.
# Fetch and byte-verify the public frozen admission source BEFORE costly setup.
source=urllib.request.urlopen('https://raw.githubusercontent.com/neverhuman/Joxing/'+p['commit']+'/scripts/host_admission.py',timeout=25).read()
d=protocol['source_files']['scripts/host_admission.py'];assert len(source)==d['bytes'] and hashlib.sha256(source).hexdigest()==d['sha256']
namespace={'__name__':'joxing_admission_frozen'};exec(compile(source,'public-host-admission.py','exec'),namespace)
policy=protocol['admission_policy'];samples=[]
try:
 samples.append(namespace['snapshot']());time.sleep(policy['spacing_s']);samples.append(namespace['snapshot']())
 admission=namespace['assess_pair'](samples,policy,time.monotonic())
except (OSError,ValueError) as error:
 admission={'status':'deferred','reasons':['snapshot_error:'+str(error)],'samples':samples,'policy':policy}
if admission['status']!='admitted':
 print(json.dumps({'status':'deferred','source_commit':p['commit'],'protocol_sha256':hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),'admission':admission,'native_worlds':'not_run','root_created':False,'epoch_started_monotonic_s':epoch,'original_epoch_elapsed_s':time.monotonic()-epoch,'bootstrap_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}));raise SystemExit(0)

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
reuse=root.exists();assert not reuse,'H29 requires a new immutable checkout'
if reuse:
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==p['commit']
 assert not (root/'runs/v0.0.11/portable-source-deployment.json.gz').exists(),'Already launched epoch'
else:root.mkdir(exist_ok=False)
commands=[['git','init',str(root)],['git','remote','add','origin','https://github.com/neverhuman/Joxing.git'],['git','fetch','origin',p['commit']],['git','checkout','--detach','FETCH_HEAD'],['git','submodule','update','--init','--depth','1','third_party/ms-human-700','third_party/msgym']]
for i,cmd in enumerate([] if reuse else commands):
 r=subprocess.run(cmd,cwd=None if i==0 else root,text=True,capture_output=True,timeout=max(1,50-(time.monotonic()-epoch)))
 if r.returncode:raise RuntimeError(r.stdout+r.stderr)
assert subprocess.check_output(['git','rev-parse','--is-shallow-repository'],cwd=root,text=True).strip()=='false'
for obj,expected in protocol['required_git_objects'].items():
 assert hashlib.sha256(subprocess.check_output(['git','show',obj],cwd=root)).hexdigest()==expected


checks=[]
for name,d in {**protocol['source_files'],**protocol['input_files'],**protocol['runtime_file_hashes']}.items():
 f=Path(name) if name.startswith('/') else root/name
 assert f.stat().st_size==d['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==d['sha256'],name
 checks.append({'path':name,**d})
fresh=namespace['snapshot']();fresh_assessed=time.monotonic();bad=namespace['reasons'](fresh,policy,fresh_assessed)
if bad:
 print(json.dumps({'status':'deferred_after_checkout','source_commit':p['commit'],'protocol_sha256':hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),'root':str(root),'admission':admission,'post_checkout':{'sample':fresh,'reasons':bad,'assessed_monotonic_s':fresh_assessed},'native_worlds':'not_run','root_created':True,'epoch_started_monotonic_s':epoch,'original_epoch_elapsed_s':time.monotonic()-epoch,'bootstrap_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}));raise SystemExit(0)
sha=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
assert time.monotonic()-epoch<50,'Original setup60s first limit with runtime reserve'
cmd=['python3','scripts/qualify_linux_fixture_readiness.py','--packet','experiments/v0.0.11/portable-linux-source-readiness-xbabe1-protocol.json','--protocol-sha256',sha,'--source-sha',p['commit'],'--epoch-start-monotonic',str(epoch)]
run=root/'runs/v0.0.11';run.mkdir(parents=True,exist_ok=True)
with (run/'portable-source-controller.log').open('x') as log:
 proc=subprocess.Popen(cmd,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()
report={'schema':'joxing-portable-source-deployment-v1','status':'launched','source_commit':p['commit'],'protocol_sha256':sha,'root':str(root),'public_files':public,'verified_files':checks,'process':{'pid':proc.pid,'start_ticks':int(st[19])},'command':cmd,'checkout_commands':commands,'load':os.getloadavg(),'available_memory_bytes':available,'free_disk_bytes':shutil.disk_usage(root).free,'other_owned_jobs':jobs,'fixture_scope':'Exact8 original bytes hydrated only under new owned output; no host aliases','epoch_started_monotonic_s':epoch,'authorized_physics_limit':protocol['physics_limit'],'prospective_rollout_transitions':0,'prospective_updates':0,'prospective_formal_probes':0,'admission':admission,'prior_admission_receipts':p['prior_admission_receipts'],'post_checkout_admission':fresh,'post_checkout_assessed_monotonic_s':fresh_assessed,'reused_unlaunched_checkout':reuse,'bootstrap_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'bootstrap_elapsed_s':time.monotonic()-epoch,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
import sys
sys.path.insert(0,str(root/'scripts'))
from receipt_encoding import V2,encode_bootstrap,compact_bytes
assert protocol['receipt_wire_encoding']==V2
blob,metadata=encode_bootstrap(report)
wire=run/'portable-source-deployment.json.gz'
temporary=wire.with_suffix(wire.suffix+'.tmp');temporary.write_bytes(blob);temporary.replace(wire)
(run/'portable-source-deployment.encoding.json').write_bytes(compact_bytes(metadata))
print(json.dumps(report))
"""
remote = remote.replace(
    "PAYLOAD", repr(base64.b64encode(json.dumps(payload).encode()).decode())
)
Path("runs/v0.0.11").mkdir(parents=True, exist_ok=True)
r = subprocess.run(
    ["ssh", "-o", "ConnectTimeout=15", packet["source_host"], "python3 -"],
    input=remote,
    text=True,
    capture_output=True,
    timeout=115,
    check=False,
)
Path("runs/v0.0.11").mkdir(parents=True, exist_ok=True)
Path("runs/v0.0.11/portable-source-xbabe1-deploy-output.txt").write_text(r.stdout + r.stderr)
print(r.stderr)
r.check_returncode()
report = json.loads(r.stdout)
history = prior + [
    report
    if report["status"] != "launched"
    else {
        k: report[k]
        for k in ("status", "source_commit", "protocol_sha256", "admission", "utc")
    }
]
history_bytes = (
    json.dumps(history, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
).encode()
if len(history_bytes) > 16384:
    Path("artifacts/v0.0.11/development/h29-xbabe1-admission-overflow-last.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    raise RuntimeError(
        "Admission history first limit; last full receipt retained separately, no further check"
    )
history_path.write_bytes(history_bytes)
if report["status"] == "launched":
    Path("artifacts/v0.0.11/development/portable-source-xbabe1-deployment-01.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "status",
                    "source_commit",
                    "protocol_sha256",
                    "root",
                    "process",
                    "load",
                    "available_memory_bytes",
                    "free_disk_bytes",
                    "other_owned_jobs",
                    "utc",
                ]
            }
        )
    )
else:
    print(json.dumps(report))
