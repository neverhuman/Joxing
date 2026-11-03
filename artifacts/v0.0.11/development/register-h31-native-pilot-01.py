"""Freeze only prospective H31protocol; no native work is launched here."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

root=Path.cwd()
old=json.loads((root/"experiments/v0.0.11/native-pair-mac-prototype-protocol.json").read_text())
source=ET.parse(root/"runs/v0.0.11/h31-pilot-guarded-dev-04/source.xml").getroot()
cases=source.findall(".//testcase")
assert len(cases)==99 and all(c.find("failure") is None and c.find("error") is None and c.find("skipped") is None for c in cases)
nodes=sorted(c.attrib["classname"].replace(".","/")+".py::"+c.attrib["name"] for c in cases)
names=set(old["source_files"])
for folder in ("src","scripts","tests"):
 names.update(str(p.relative_to(root)) for p in (root/folder).rglob("*.py"))
def receipt(path):
 p=Path(path);p=p if p.is_absolute() else root/p
 return {"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
p={k:old[k] for k in ("runtime_python","runtime_versions","runtime_file_hashes","input_files","upstream_commits","minimum_disk_bytes","minimum_memory_bytes","teacher_inputs","model","objective","actor_normalization","source_host")}
p.update(schema="joxing-native-pair-motor-pilot-v1",scope="Explicit local Mac two-seed motor-learning behavior pilot, NOTcompetitive boxing/self-play qualification",
 deployment_root="/Users/bentaylor/joxing/native-pair-motor-pilot-20261103-01",output="runs/v0.0.11/native-pair-motor-pilot-01",
 source_files={n:receipt(n) for n in sorted(names)},test_node_ids=nodes,
 test_modules=["tests/test_native_pair_motor_rl.py","tests/test_native_pair_motor_pilot.py","tests/test_native_pair_guarded_stage.py","tests/test_native_pair_motor_evidence.py","tests/test_native_pair_motor_vec.py","tests/test_history.py","tests/test_rules.py","tests/test_match_referee.py","tests/test_native_tape_storage.py"],
 simulation_workers=1,threads_per_process=1,process_ceiling=12,owned_rss_bytes=8*1024**3,
 whole_wall_s=1800,setup_wall_s=60,source_wall_s=120,native_wall_s=600,render_wall_s=30,audit_wall_s=300,
 raw_bytes=1024**3,receipt_bytes=16*1024**2,physics_steps=200000,source_fixture_physics_reserve=2048,
 source_fixture_physics_scope="Fixed successful source fixture loops1243; conservative2048reserve, not measured. All99fresh controls required before actual checkpoint work.",
 seeds=[761,983],joint_steps=1024,horizon_ticks=3000,evaluation_seeds=[71000,71016,71032,71048],
 expected_max_primary_physics=89197,expected_max_independent_audit_physics=88960,expected_max_reserved_physics=180205,
 learner={"algorithm":"SB3PPO CPU","n_steps":128,"batch_size":64,"n_epochs":4,"learning_rate":.0003,"gamma":.99,"net_arch":[64,64],"log_std_init":-3.0,"ortho_init":False,"initial_actor_mean_zero":True,"one_shared_policy_two_simultaneous_logical_slots":True},
 evidence={"chunk_rows":128,"chunk_max_bytes":64*1024**2,"row_text_max_bytes":3072,"full_checkpoints_per_seed":9,"checkpoint_max_bytes":13*1024**2,"continuation_max_bytes":1024**2,"other_raw_reservation_bytes":240*1024**2,"other_text_reservation_bytes":8*1024**2,"all_training_steps_native_replay":True,"all16evaluations_exact_replay":True,"fixed_clips":"First entire reset71000before/after per seed,4clips, falls and0points included; no render physics","durable_primary_audit_counters":"16byte count+bitwise complement, updated BEFORE every entered interval; source reserved separately","guard":"Checked owned Python writes+boundedTorch buffer; C/ffmpeg RLIMIT_FSIZE; serial source/media reserves and captured logs; sampled8GiBRSS/process/wholewall owner. No Linux100mslifetime or adversarial OS-write guarantee claimed."},
 policy_recompute_atol=.00001,
 behavioral_criteria={"per_seed_per_side_all_required":True,"mean_target_error_improvement_m_min":.05,"mean_survival_change_s_min":-.1,"after_full_cycle_starts_min_of4":2,"after_completed3s_starts_min_of4":3,"cycle":{"max_both_glove_target_distance_m":.15,"right_glove_face_cos_min":.5,"continuous_physics_ticks_per_phase":40,"required_phase_order":[0,1,2],"cycle_must_complete_within_one1000tick_schedule":True,"measurement":"Native glove centers/goals every1ms; no endpoint-only persistence inference"}},
 conclusion="Separate audit execution correctness from behavioral pass/fail. Actor change/return/video alone is not improvement. Preserve both seeds, all16evals, invalid/not_run/falls/ties. Competitive boxing and historical-opponent league remain unqualified.",
 limitations=["Explicit Mac profile, not Linux arithmetic/parity/native100ms retirement qualification","Only2x1024joint steps; no1k/100kgamecount claim","Zero-delay causal4frame history; jitter queues unqualified","Same named motor objective/138residual groups/.1amplitude/all700muscles per actor; no physics/rules edits","Audit stage300s prospectively for88960physical replay intervals versus old640interval witness; no prior failed audit budget is extended","Other evidence reservation240MiB allocates source64/evals100/media64/fault12; 4Cencoders each4MiBfile cap and16MiBstage reserve, all within same1GiBraw hard ceiling"],
 prior_costs={"actual_prototype":"32jointsteps/64samples/4updates, fixed1supright0-0, noimprovement","streaming":"55+39sourcecontrols,43.3835%smaller exactfixture evidence","pilot_guarded_development":"dev03:97/99,checked-file rejected by SB3open_path; fixed via its public singledispatch registration, dev04:99/99"})
path=root/"experiments/v0.0.11/native-pair-motor-pilot-01.json";assert not path.exists();path.write_text(json.dumps(p,indent=2,sort_keys=True)+"\n")
print(json.dumps({"protocol":str(path),"sha256":hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(),"source_files":len(p["source_files"]),"fresh_cases":len(nodes)}))
