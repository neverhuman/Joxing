"""Read-only POST observation; never proves conditions during an earlier fault."""
import json,subprocess,sys
from pathlib import Path
code = r"""
import json,os,time
from pathlib import Path
r={'scope':'POST-failure read-only host observation, not in-import evidence','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'load':os.getloadavg(),'cpu_count':os.cpu_count(),'affinity_cpus':len(os.sched_getaffinity(0))}
for n in ['/proc/pressure/cpu','/proc/pressure/io','/proc/pressure/memory','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/cpu.stat']:
 p=Path(n);r[n]=p.read_text() if p.exists() else None
jobs=[]
for d in Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:
  if str((d/'cwd').resolve()).startswith('/home/ubuntu/joxing/') and b'python' in (d/'cmdline').read_bytes():jobs.append(int(d.name))
 except OSError:pass
r['owned_jobs']=jobs
print(json.dumps(r))
"""
r=subprocess.run(['ssh','xbabe1','python3 -'],input=code,text=True,capture_output=True,timeout=20)
r.check_returncode();data=json.loads(r.stdout)
Path(sys.argv[1]).write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({'utc':data['utc'],'load':data['load'],'owned_jobs':data['owned_jobs'],'scope':data['scope']}))
