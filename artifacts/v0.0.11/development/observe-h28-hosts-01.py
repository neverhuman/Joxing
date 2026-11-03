import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

output = Path(sys.argv[1])
assert not output.exists(), "Fresh inventory path required"
linux = r"""
import os,json,time,shutil
from pathlib import Path

def sample():
 jobs=[]
 for d in Path('/proc').iterdir():
  if not d.name.isdigit():continue
  try:
   cwd=str((d/'cwd').resolve());cmd=(d/'cmdline').read_bytes()
   if cwd.startswith('/home/ubuntu/joxing/') and (b'python' in cmd or b'/uv ' in cmd):jobs.append({'pid':int(d.name),'command':cmd.replace(bytes([0]),b' ').decode(errors='replace')[:240]})
  except OSError:pass
 return {'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'monotonic_s':time.monotonic(),'load':os.getloadavg(),'affinity':len(os.sched_getaffinity(0)),'available_memory_bytes':int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024,'workspace_exists':Path('/home/ubuntu/joxing').exists(),'disk_free_bytes':shutil.disk_usage('/home/ubuntu/joxing' if Path('/home/ubuntu/joxing').exists() else Path.home()).free,'psi':{k:Path('/proc/pressure',k).read_text() for k in ('cpu','io','memory')},'jobs':jobs}
a=sample();time.sleep(10);b=sample();print(json.dumps({'samples':[a,b]}))
"""
mac = r"""
import subprocess,json,os,time
from pathlib import Path
p=subprocess.check_output(['ps','-axo','pid=,command='],text=True)
jobs=[x.strip() for x in p.splitlines() if '/joxing/' in x and ('python' in x or '/uv ' in x) and 'ps -axo' not in x]
print(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'load':os.getloadavg(),'jobs':jobs,'vm_stat':subprocess.check_output(['vm_stat'],text=True),'cpu_count':os.cpu_count()}))
"""


def collect(host):
    r = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", host, "python3 -"],
        input=mac if host == "xbabe4" else linux,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return {
        "host": host,
        "returncode": r.returncode,
        "result": json.loads(r.stdout) if r.returncode == 0 else None,
        "stderr": r.stderr,
    }


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    rows = list(pool.map(collect, ["xbabe1", "xbabe2", "xbabe3", "xbabe4"]))
value = {"scope": "Read-only fresh capacity; no reservation/native work", "hosts": rows}
with output.open("x") as stream:
    stream.write(json.dumps(value, indent=2) + "\n")
for r in rows:
    v = r["result"]
    last = v.get("samples", [v])[-1] if v else {}
    print(
        json.dumps(
            {
                "host": r["host"],
                "returncode": r["returncode"],
                **{
                    k: last[k]
                    for k in ("load", "available_memory_bytes", "jobs", "psi")
                    if k in last
                },
            }
        )
    )
