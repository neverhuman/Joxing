"""Read-only source-host reproduction. Run from the frozen ef42c87 checkout.

uv run --project experiments/v0.0.11/runtime --locked python /path/to/this.py
No new physics, writes, rerun, source mutation or full-D1success declaration.
"""
import json
from pathlib import Path
from joxing.sensor_protocol import load_scene, verify_inputs
from joxing.sensor_physics_audit import audit_sensor_physics
from joxing.sensor_audit import compare_sensor_replay, compare_zero_delay_oracle
from joxing.boxing_audit import require
root=Path.cwd();out=root/"runs/v0.0.11/sensor-proof-01"
p=json.loads((out/"protocol.json").read_text());r=json.loads((out/"result.json").read_text())
require(r["source_commit"]=="ef42c872d13bfebb16b347f842ba11f0b8b8e776", "Foreign source")
require(r["status"]=="halted" and r["physics_attempted"]==2431, "Partial denominator")
require([x["status"] for x in r["rows"]]==["completed"]*6+["incomplete","not_run"],"Attempt denominator")
verify_inputs(p,root);scene=load_scene(p,root)
report={"status":"partial_evidence_passed","whole_D1_passed":False,"new_physics":0,"cases":[],"replays":[]}
for c in p["cases"][:6]:
 report["cases"].append({"id":c["id"],**audit_sensor_physics(out/c["id"],scene,expected_physics=c["physics"],delays=p["delays_ticks"])})
 if "parent" in c:report["replays"].append({"id":c["id"],**compare_sensor_replay(out/c["parent"],out/c["id"],source_checkpoint=out/c["parent"]/(c["checkpoint"]+".json"))})
report["oracle"]=compare_zero_delay_oracle(Path(p["oracle_root"])/"source-90101",out/"zero-90101")
print(json.dumps(report,indent=2))
