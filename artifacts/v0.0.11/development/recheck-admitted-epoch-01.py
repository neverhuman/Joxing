"""Recheck ONE frozen epoch using an authentic sparse Git source checkout.
Does not change source/protocol/policy or create an additional simulation epoch.
"""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
SOURCE='1c9415240368d78defe5d85f7367150d6c6434fd'
PROTOCOL='fbb5961fb9c82fc28e581145b3e1d99591d612dbf4c7560fb677c6943f2566c1'
ROOT=Path(__file__).resolve().parents[3]
PACKET='experiments/v0.0.11/resource-admitted-four-context-protocol.json'
HELPER='artifacts/v0.0.11/development/deploy-resource-admitted-four-context-01.py'
HISTORY='artifacts/v0.0.11/development/h23-admission-history.json'
STAGING=ROOT/'runs/v0.0.11/frozen-admission-1c94152'
parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
def git(*parts,cwd=ROOT,**kw):return subprocess.check_output(['git',*parts],cwd=cwd,**kw)
p=json.loads(git('show',SOURCE+':'+PACKET))
assert hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==PROTOCOL
names=[n for n in p['source_files'] if not n.startswith('third_party/')]+[PACKET,'docs/RESOURCE_ADMISSION_MISSION.md','research/README.md','research/H23.md','docs/FOUR_NATIVE_CONTEXT_RESULTS.md']
if not STAGING.exists():
 STAGING.parent.mkdir(parents=True,exist_ok=True)
 git('clone','--shared','--no-checkout',str(ROOT),str(STAGING))
 git('sparse-checkout','set','--no-cone','--stdin',cwd=STAGING,input=('\n'.join('/'+n for n in names)+'\n').encode())
 git('checkout','--detach',SOURCE,cwd=STAGING)
assert git('rev-parse','HEAD',cwd=STAGING,text=True).strip()==SOURCE
assert not git('diff','--name-only','HEAD',cwd=STAGING,text=True).strip()
for name in names:
 assert (STAGING/name).read_bytes()==git('show',SOURCE+':'+name),name
prior=json.loads((ROOT/HISTORY).read_text()) if (ROOT/HISTORY).exists() else []
if any(x.get('status')=='launched' for x in prior):
 print(json.dumps({'status':'already_launched_inspect_owned_task','source_commit':SOURCE}));raise SystemExit(0)
(STAGING/HISTORY).parent.mkdir(parents=True,exist_ok=True)
(STAGING/HISTORY).write_text(json.dumps(prior,sort_keys=True,separators=(',',':'))+'\n')
if args.prepare_only:
 print(json.dumps({'status':'prepared_frozen_epoch','source_commit':SOURCE,'public_files_verified_locally':len(names),'admission_records':len(prior),'simulation_launched':False}));raise SystemExit(0)
(STAGING/'runs/v0.0.11').mkdir(parents=True,exist_ok=True)
result=subprocess.run([sys.executable,HELPER],cwd=STAGING,text=True,capture_output=True,timeout=420)
(ROOT/'runs/v0.0.11/frozen-admission-recheck-output.txt').write_text(result.stdout+result.stderr)
if (STAGING/HISTORY).exists():
 after=json.loads((STAGING/HISTORY).read_text())
 canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'))
 assert canonical(after[:len(prior)])==canonical(prior),'Admission history changed'
 assert len(after)>=len(prior)
 assert canonical(json.loads((ROOT/HISTORY).read_text()) if (ROOT/HISTORY).exists() else [])==canonical(prior),'Concurrent admission writer'
 shutil.copyfile(STAGING/HISTORY,ROOT/HISTORY)
receipt='artifacts/v0.0.11/development/resource-admitted-four-context-deployment-01.json'
if (STAGING/receipt).exists():shutil.copyfile(STAGING/receipt,ROOT/receipt)
print(result.stdout);print(result.stderr,file=sys.stderr);result.check_returncode()
