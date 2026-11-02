"""Preserve symlink metadata; transfer only dereferenced regular owned file bytes."""
import base64,hashlib,io,json,shlex,subprocess,tarfile,gzip
from pathlib import Path
p=json.loads(Path('experiments/v0.0.11/resource-admitted-four-context-protocol.json').read_text());audit_path=Path('runs/v0.0.11/resource-admitted-native-01-integrity.json');audit_raw=audit_path.read_bytes() if audit_path.exists() else gzip.decompress(Path('artifacts/v0.0.11/resource-admitted-native-01/independent-audit.json.gz').read_bytes());a=json.loads(audit_raw)
remote=r"""
from pathlib import Path
import os,json
root=Path(ROOT)
rows=[]
for f in (root/'runs/v0.0.11').rglob('*'):
 if f.is_symlink():
  assert f.resolve().is_relative_to(root),str(f)
  rows.append({'path':str(f.relative_to(root)),'link_target':os.readlink(f),'resolved_within_owned_root':True,'is_file':f.is_file()})
print(json.dumps(rows))
""".replace('ROOT',repr(p['deployment_root']))
r=subprocess.run(['ssh','xbabe1','python3 -'],input=remote,text=True,capture_output=True,timeout=20);r.check_returncode();links=json.loads(r.stdout)
argv=['tar','--dereference','--hard-dereference','-C',p['deployment_root'],'-cf','-','--',*a['raw_manifest']]
blob=subprocess.check_output(['ssh','xbabe1',shlex.join(argv)],timeout=60)
archive=Path('runs/v0.0.11/resource-admitted-native-01-owned-bytes.tar');assert not archive.exists();archive.write_bytes(blob)
out=Path('runs/v0.0.11/resource-admitted-native-01-safe-evidence');out.mkdir(exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(blob)) as t:
 members=t.getmembers();assert len(members)==len(a['raw_manifest']) and {m.name for m in members}==set(a['raw_manifest']) and all(m.isfile() for m in members)
 t.extractall(out,filter='data')
for n,d in a['raw_manifest'].items():
 f=out/n;assert f.is_file() and not f.is_symlink() and f.stat().st_size==d['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==d['sha256'],n
curated=Path('artifacts/v0.0.11/resource-admitted-native-01');curated.mkdir(exist_ok=False)
report=json.loads(gzip.decompress((out/p['output']/'result.json.gz').read_bytes()))
(curated/'controls.xml').write_bytes(gzip.decompress((out/p['output']/'controls.xml.gz').read_bytes()))
facts=report['rows'][0]['facts'];w=next(w for w in a['watchdog_reports'] if w['path'].endswith('native0-watchdog.json'));keys={f"{v['pid']}:{v['start']}" for v in facts['workers']};native_retirements={k:v for k,v in w['retirement_records'].items() if k in keys};assert len(native_retirements)==4
transfer={'status':'passed_exact_owned_file_bytes','raw_files':len(a['raw_manifest']),'raw_bytes':sum(d['bytes'] for d in a['raw_manifest'].values()),'tar_sha256':hashlib.sha256(blob).hexdigest(),'all_names_sizes_hashes_verified':True,'original_symlink_metadata':links,'representation':'regular file payloads for safe extraction; original symlink targets retained separately','original_safe_filter_fault':'Initial archive extraction stopped at absolute test-owned symlink, correctly refused by tar data filter. Partial safe prefix retained separately; source unchanged, no simulation rerun.'}
(curated/'transfer.json').write_text(json.dumps(transfer,indent=2)+'\n')
summary={'status':'independently_qualified_four_native_contexts','source_commit':a['source_commit'],'protocol_sha256':a['protocol_sha256'],'source_files_verified':a['source_files_verified'],'historical_objects_verified':a['historical_objects_verified'],'complete_superproject_history':a['complete_superproject_history'],'controls':a['counts'],'native_packet':{'attempted':1,'valid':1,'invalid':0,'not_run':0},'native_worlds':{'attempted':4,'valid':4,'invalid':0,'not_run':0},'state_hash':p['expected_state_hash'],'specification':p['expected_specification'],'preparation_steps':800,'rollout_steps':0,'policy_inferences':0,'learner_updates':0,'actors_constructed':0,'formal_boxing_games':0,'close_to_join_s':facts['close_to_join_s'],'wall_s':report['wall_s'],'peak_capacity_charge_bytes':w['peak_capacity_charge_bytes'],'peak_processes':w['peak_processes'],'max_sample_gap_s':w['max_sample_gap_s'],'max_probe_s':w['max_probe_s'],'native_retirement_records':native_retirements,'workers':facts['workers'],'physics_credit':a['credit_audit'],'encoded_receipts':a['encoded_deployment'],'fixture_bytes':a['fixture_bytes'],'receipt_bytes':a['receipt_bytes'],'exited_recorded_identities':a['exited_recorded_identities'],'live_writers':a['live_writers'],'raw_manifest_sha256':hashlib.sha256(json.dumps(a['raw_manifest'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),'full_independent_audit_sha256':hashlib.sha256(audit_raw).hexdigest(),'raw_files':len(a['raw_manifest']),'raw_bytes':transfer['raw_bytes'],'scope':'Four original700muscle/200prep constructors, exact native state/lifetime/retirement; no learning, stable guard, boxing or native performance comparison'}
(curated/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(curated/'constructor-result.json').write_text(json.dumps(facts,indent=2)+'\n')
Path('artifacts/v0.0.11/development/resource-admitted-native-01-transfer.json').write_text(json.dumps({k:v for k,v in transfer.items() if k!='original_symlink_metadata'},indent=2)+'\n')
print(json.dumps({'status':summary['status'],'native_worlds':summary['native_worlds'],'close_s':summary['close_to_join_s'],'hold_ms':[1000*v['max_hold_s'] for v in native_retirements.values()],'raw_files':transfer['raw_files'],'raw_bytes':transfer['raw_bytes'],'retained_symlinks':len(links),'curated_summary_bytes':(curated/'summary.json').stat().st_size}))
