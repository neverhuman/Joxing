"""Read-only four-host capacity inventory; never schedules or reserves work."""

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path

REMOTE = r'''
import json,os,platform,re,shutil,subprocess,time
from pathlib import Path

def command(args):
    try:
        p=subprocess.run(args,text=True,capture_output=True,timeout=8)
        return {'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
    except (OSError,subprocess.TimeoutExpired) as e:
        return {'error':type(e).__name__}

def snapshot():
    result={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'monotonic_s':time.monotonic(),'load':list(os.getloadavg()),
            'cpu_count':os.cpu_count(),'disk_free_bytes':shutil.disk_usage(Path.home()).free}
    if platform.system()=='Linux':
        result['affinity_cpus']=len(os.sched_getaffinity(0))
        m={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.split(':')[0] in ('MemAvailable','MemTotal','SwapFree','SwapTotal')}
        result['memory_bytes']=m
        result['pressure_raw']={k:Path('/proc/pressure',k).read_text() for k in ('cpu','io','memory')}
        result['cpu_ticks']=[int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
        jobs=[]
        for d in Path('/proc').iterdir():
            if not d.name.isdigit() or int(d.name)==os.getpid():continue
            try:
                cmd=(d/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
                cwd=str((d/'cwd').resolve())
                if (cwd.startswith(str(Path.home()/'joxing')+'/') or str(Path.home()/'joxing')+'/' in cmd) and any(k in cmd for k in ('python','uv run','qualify','pytest')):
                    s=(d/'stat').read_text().rsplit(')',1)[1].split()
                    jobs.append({'pid':int(d.name),'start_ticks':int(s[19]),'state':s[0],'cwd':cwd,'command':cmd[:350]})
            except (OSError,ValueError):pass
        result['joxing_jobs']=jobs
    else:
        result['memory_total_bytes']=int(command(['sysctl','-n','hw.memsize'])['stdout'])
        result['vm_stat']=command(['vm_stat'])
        result['memory_pressure_query']=command(['memory_pressure','-Q'])
        # A free/reclaimable-page estimate is NOT Linux MemAvailable.
        raw=result['vm_stat'].get('stdout','')
        page=re.search(r'page size of (\d+) bytes',raw)
        counts=dict(re.findall(r'^(Pages [^:]+):\s+(\d+)\.',raw,re.M))
        if page:
            result['free_inactive_speculative_bytes_estimate']=int(page.group(1))*sum(int(counts.get(k,0)) for k in ('Pages free','Pages inactive','Pages speculative'))
        result['cpu_activity']=command(['top','-l','1','-n','0'])
        ps=command(['ps','-axo','pid=,stat=,pcpu=,rss=,command='])
        result['joxing_jobs']=[line.strip()[:450] for line in ps.get('stdout','').splitlines() if '/joxing/' in line.lower() and ('python' in line or 'pytest' in line)]
    return result

first=snapshot();time.sleep(10);second=snapshot()
report={'platform':platform.system(),'machine':platform.machine(),'python_version':platform.python_version(),
        'samples':[first,second],'interval_s':second['monotonic_s']-first['monotonic_s'],
        'tools':{x:shutil.which(x) is not None for x in ('uv','docker','nvidia-smi')},
        'scope':'read-only capacity; no reservation, benchmark, migration or qualification'}
if platform.system()=='Linux':
    a=first['cpu_ticks'];b=second['cpu_ticks'];delta=[y-x for x,y in zip(a,b)]
    report['cpu_busy_percent_over_interval']=100*(1-(delta[3]+delta[4])/sum(delta)) if sum(delta)>0 else None
    if shutil.which('nvidia-smi'):
        report['gpu']=command(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'])
print(json.dumps(report,sort_keys=True))
'''


def inspect(host):
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", host, "python3 -"],
        input=REMOTE,
        text=True,
        capture_output=True,
        timeout=65,
        check=False,
    )
    if result.returncode:
        return {"host": host, "error": result.stderr.strip()[:500]}
    return {"host": host, **json.loads(result.stdout)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Refusing to overwrite inventory")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        reports = list(pool.map(inspect, ("xbabe1", "xbabe2", "xbabe3", "xbabe4")))
    report = {
        "schema": "joxing-four-host-capacity-v1",
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "snapshots": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    for row in reports:
        if "error" in row:
            print(json.dumps(row))
            continue
        last = row["samples"][-1]
        print(json.dumps({"host": row["host"], "platform": row["platform"],
                          "load": last["load"],
                          "cpu_busy_percent": row.get("cpu_busy_percent_over_interval"),
                          "memory": last.get("memory_bytes", last.get("free_inactive_speculative_bytes_estimate")),
                          "pressure": last.get("pressure_raw"),
                          "owned_jobs": last["joxing_jobs"], "gpu": row.get("gpu")}))


if __name__ == "__main__":
    main()
