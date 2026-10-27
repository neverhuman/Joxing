import os,json,pathlib,subprocess,time,signal,traceback,datetime
root=pathlib.Path('/home/ubuntu/joxing/v009-collector-20261027');os.chdir(root);out=root/'runs/v0.0.9'
r={'pid':os.getpid(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'status':'running','stages':[],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'physics_budget':100000,'threads':1}
def save():
 p=out/'supervisor.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2)+'\n');t.replace(p)
commands=[('sync', ['/home/ubuntu/.local/bin/uv', 'sync', '--project', 'experiments/v0.0.9/runtime', '--locked'], 180), ('full-tests', ['/home/ubuntu/.local/bin/uv', 'run', '--project', 'experiments/v0.0.9/runtime', '--locked', 'pytest', '-q'], 360), ('H3', ['/home/ubuntu/.local/bin/uv', 'run', '--project', 'experiments/v0.0.9/runtime', '--locked', 'python', 'scripts/benchmark_collector.py', '--protocol', 'experiments/v0.0.9/collector-protocol.json', '--inputs', '/home/ubuntu/joxing/deployment-cache/inputs', '--out', 'runs/v0.0.9/collector-01'], 250)]
try:
 for name,cmd,limit in commands:
  row={'stage':name,'command':cmd,'status':'running'};r['stages'].append(row);save();start=time.monotonic()
  with (out/(name+'.log')).open('w') as log:
   p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=p.pid;save()
   try:rc=p.wait(timeout=limit)
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    raise
  row.update(returncode=rc,status='completed' if rc==0 else 'failed',wall_s=time.monotonic()-start);save()
  if rc:raise RuntimeError(name+' returned '+str(rc))
  if name=='full-tests' and '165 passed' not in (out/(name+'.log')).read_text():raise RuntimeError('Expected all165Linux regressions before H3')
 r['status']='completed'
except Exception as error:r.update(status='failed',error=str(error),traceback=traceback.format_exc())
finally:r['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
