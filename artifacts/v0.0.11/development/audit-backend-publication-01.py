import base64,hashlib,io,json,subprocess,sys,tarfile,shutil
from pathlib import Path
packet_name,label=sys.argv[1:];p=json.loads(Path(packet_name).read_text())
remote=r"""
import base64,hashlib,json,subprocess,time,xml.etree.ElementTree as ET
from pathlib import Path
p=json.loads(base64.b64decode(PAYLOAD));root=Path(p['deployment_root']);out=root/p['output'];r=json.loads((out/'result.json').read_text());assert r['status'] in ('halted','completed')
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as s:
  while b:=s.read(1048576):h.update(b)
 return h.hexdigest()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==r['source_commit'];assert not subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=root,text=True).strip();assert subprocess.check_output(['git','rev-parse','--is-shallow-repository'],cwd=root,text=True).strip()=='false'
for n,d in p['source_files'].items():
 f=root/n;assert f.stat().st_size==d['bytes'] and sha(f)==d['sha256'],n
for obj,h in p['required_git_objects'].items():assert hashlib.sha256(subprocess.check_output(['git','show',obj],cwd=root)).hexdigest()==h
mapping={}
for node in p['test_node_ids']:
 a=node.split('::');mapping[(a[0][:-3].replace('/','.')+''.join('.'+x for x in a[1:-1]),a[-1])]=node
cases=[]
for c in ET.parse(out/'controls.xml').iter('testcase'):
 key=c.get('classname'),c.get('name');assert key in mapping
 status='failed' if c.find('failure') is not None or c.find('error') is not None else 'skipped' if c.find('skipped') is not None else 'passed'
 cases.append({'node':mapping[key],'status':status})
assert {c['node'] for c in cases}==set(p['test_node_ids']) and len(cases)==len(p['test_node_ids']) and all(c['status']=='passed' for c in cases)
ids=[r['supervisor']]+r.get('owned_identities',[])
for s in r['stages']:
 if s.get('process'):ids.append(s['process'])
 ids.extend(s.get('owned_identities',[]))
watchdogs=[]
for f in sorted(set(out.glob('*-watchdog.json')) | set(out.rglob('watchdog.json'))):
 try:w=json.loads(f.read_text())
 except json.JSONDecodeError:
  assert '/test_private_scope_' in str(f),str(f);continue
 if 'root' not in w:
  assert '/test_private_scope_' in str(f),str(f);continue
 ids += [w['root'],w['supervisor'],w['watchdog']]+w['owned_identities']
 watchdogs.append({'path':str(f.relative_to(root)),**w})
credit_path=out/'fixtures/native-credit.json'
credit=json.loads(credit_path.read_text()) if credit_path.exists() else None
credit_events=[]
if credit:
 lines=credit_path.with_suffix('.jsonl').read_text().splitlines()
 used=0;pending={};chain='0'*64;settled={}
 for n,line in enumerate(lines[:credit['sequence']],1):
  e=json.loads(line);assert e['sequence']==n
  ids.append({'pid':e['owner']['pid'],'start_ticks':int(e['owner']['start'])})
  chain=hashlib.sha256((chain+line).encode()).hexdigest()
  token=e['token'];op=e['operation']
  if op=='reserve':
   assert token not in pending and e['amount']>0
   pending[token]={'owner':e['owner'],'reserved':e['amount'],'slot':e['slot'],'kind':e['kind'],'state':'reserved'}
  elif op=='commit':
   x=pending[token];assert x['owner']==e['owner'] and x['state']=='reserved' and 0<=e['actual']<=x['reserved']
   used+=e['actual'];key=x['slot']+':'+x['kind'];total=settled.setdefault(key,{'operations':0,'used':0});total['operations']+=1;total['used']+=e['actual'];del pending[token]
  else:
   assert op=='unknown' and pending[token]['owner']==e['owner'];pending[token].update(state='unknown',error=e['error'])
  assert used+sum(x['reserved'] for x in pending.values())<=credit['limit']
  credit_events.append(e)
 assert used==credit['used'] and pending==credit['pending'] and chain==credit['chain'] and settled==credit['settled']
 assert credit['limit']==p['physics_limit']
 if r['status']=='completed':assert not pending and len(lines)==credit['sequence']

expected_delay_paths=set()
if p.get('watchdog_output_backend')=='private-tmpfs-v1':
 for f in out.rglob('publication-control.json'):
  c=json.loads(f.read_text())
  if c['backend']=='disk':
   assert c['expected_status']==c['actual_status']=='stopped' and c['declared_disk_delay_s']==.15
   expected_delay_paths.add(str((f.parent/'watchdog.json').relative_to(root)))
 assert len(expected_delay_paths)==1
for w in watchdogs:
 if w['path'] in expected_delay_paths:
  assert w['status']=='stopped' and w['reason']=='stale_monitor' and w['max_sample_gap_s']>.1
  assert w['max_probe_s']<=.1 and w['phase_timing']['max_wall_s']['report_write']>=.15
  assert not w['final_live'] and not w.get('pending_retirements')
  continue
 if w.get('rss_reader')!='terminal-mm-reserve-v3':continue
 if r['status']=='completed':
  assert (w['status']=='exited' or (w['status']=='stopped' and w['reason']=='planned_rss')) and not w['pending_retirements'] and w['peak_capacity_charge_bytes']<=p['owned_rss_bytes']
  assert max(w['max_probe_s'],w['max_sample_gap_s'])<=p['max_monitor_gap_s']
 for x in w['retirement_records'].values():
  e=x['evidence'];assert e['tasks']==e['after_tasks']==[str(x['pid'])] and e['statm_pages']==[0]*7 and e['status_threads']==1
  assert e['vmrss_absent'] and e['vmsize_absent'] and x['highwater_charge_bytes']>0
  assert all(z['start_ticks']==x['start_ticks'] and z['flags']&4 for z in (e['before'],e['after'],e['task']))
  if r['status']=='completed':assert x['released'] and x['max_hold_s']<=.1
 for sample in w['recent_samples']:
  assert sample['capacity_charge_bytes']==sample['observed_rss_bytes']+sample['retiring_rss_charge_bytes']
for n,d in p.get('input_files',{}).items():
 f=Path(n);assert f.stat().st_size==d['bytes'] and sha(f)==d['sha256']
constructor_facts=None
if p['schema']=='joxing-native-teardown-v1' and r['status']=='completed':
 folder=out/'fixtures/native0';constructor_facts=json.loads((folder/'constructor-result.json').read_text())
 assert constructor_facts==r['rows'][0]['facts'] and r['constructor_gate_qualified']
 assert constructor_facts['muscles']==700 and constructor_facts['preparation_steps']==200
 assert constructor_facts['actors_constructed']==constructor_facts['learner_updates']==constructor_facts['rollout_steps']==0
 assert constructor_facts['specification']==p['expected_specification'] and credit['used']==200 and not credit['pending']
 assert credit['settled']=={'native0:preparation':{'operations':1,'used':200}}
 native=[json.loads(line) for line in (folder/'traces/native-world-0.jsonl').read_text().splitlines()]
 closed=[json.loads(line) for line in (folder/'traces/worker-0.jsonl').read_text().splitlines()]
 assert len(native)==len(closed)==1 and native[0]['semantic']['phase']=='reset' and native[0]['semantic']['muscles']==700
 assert native[0]['owner']==constructor_facts['worker'] and closed[0]['semantic']['phase']=='close' and closed[0]['semantic']['state']==constructor_facts['initial_and_close_state_hash']
 assert not list((folder/'guard-events').glob('*.jsonl')) and not list(folder.rglob('*.pt')) and not list(folder.rglob('*.zip'))
 key=f"{constructor_facts['worker']['pid']}:{constructor_facts['worker']['start']}"
 assert key in watchdogs[0]['retirement_records'] and watchdogs[0]['retirement_records'][key]['released']

serial_facts=None
for n,d in p.get('runtime_file_hashes',{}).items():
 f=root/n;assert f.stat().st_size==d['bytes'] and sha(f)==d['sha256']
for f in out.rglob('foreign.json'):
 x=json.loads(f.read_text());assert x['survived_guard'];ids.append(x['identity'])
if p['schema']=='joxing-serial-native-teardown-v1' and r['status']=='completed':
 folder=out/'fixtures/native0';serial_facts=json.loads((folder/'constructor-result.json').read_text())
 assert serial_facts==r['rows'][0]['facts'] and r['constructor_gate_qualified']
 assert serial_facts['schema']==p['schema'] and serial_facts['close_transport']=='serial-close-v1'
 assert serial_facts['muscles_per_worker']==[700]*4 and serial_facts['preparation_steps']==800
 assert serial_facts['rollout_steps']==serial_facts['actors_constructed']==serial_facts['policy_inferences']==serial_facts['learner_updates']==0
 assert serial_facts['specification']==p['expected_specification'] and serial_facts['normalizer']=='none'
 assert len(serial_facts['workers'])==len(serial_facts['initial_and_close_state_hashes'])==4
 assert credit['used']==800 and not credit['pending']
 assert credit['settled']=={'native0:preparation':{'operations':4,'used':800}}
 events=[json.loads(line) for line in (folder/'serial-close.jsonl').read_text().splitlines()]
 assert len(events)==10 and events[0]['phase']=='bound' and events[-1]['phase']=='closed'
 assert [(e['phase'],e['worker']) for e in events[1:-1]]==[(phase,i) for i in range(4) for phase in ('close_send','joined')]
 assert all(a['monotonic_s']<=b['monotonic_s'] for a,b in zip(events,events[1:]))
 assert events[-1]['monotonic_s']-events[1]['monotonic_s']<=p['close_deadline_s']
 native_watch=next(w for w in watchdogs if w['path'].endswith('native0-watchdog.json'))
 native_keys={f"{w['pid']}:{w['start']}" for w in serial_facts['workers']}
 assert native_keys & set(native_watch['retirement_records'])
 assert not set(native_watch['retirement_records'])-native_keys
 for i,w in enumerate(serial_facts['workers']):
  native=[json.loads(line) for line in (folder/f'traces/native-world-{i}.jsonl').read_text().splitlines()]
  closed=[json.loads(line) for line in (folder/f'traces/worker-{i}.jsonl').read_text().splitlines()]
  assert len(native)==len(closed)==1 and native[0]['owner']==closed[0]['owner']==w
  assert native[0]['semantic']['phase']=='reset' and native[0]['semantic']['muscles']==700
  assert closed[0]['semantic']['phase']=='close' and closed[0]['semantic']['state']==serial_facts['initial_and_close_state_hashes'][i]
  assert events[0]['workers'][i]['pid']==w['pid'] and events[0]['workers'][i]['start']==w['start']
  for e in events[1+2*i:3+2*i]:assert e['identity']['pid']==w['pid'] and e['identity']['start']==w['start']
  assert events[1+2*i]['monotonic_s']<=closed[0]['monotonic_s']<=events[2+2*i]['monotonic_s']
  assert events[2+2*i]['exitcode']==0
 assert not list((folder/'guard-events').glob('*.jsonl')) and not list(folder.rglob('*.pt')) and not list(folder.rglob('*.zip'))
 assert len(watchdogs)==5
 controls=[w for w in watchdogs if not w['path'].endswith('native0-watchdog.json')]
 assert len([w for w in controls if w.get('rss_reader')=='terminal-mm-reserve-v3'])==2
 for w in controls:
  assert not w['final_live'] and max(w['max_sample_gap_s'],w['max_probe_s'])<=.1
  assert w['peak_sampled_rss_bytes']<=w['limits']['hard_rss']
  if w.get('rss_reader')=='terminal-mm-reserve-v3':assert w['retirement_records']
  if w['status']=='stopped':assert w['reason']=='planned_rss'
  else:assert w['status']=='exited'


release_facts=None
if p['schema']=='joxing-release-native-teardown-v1' and r['status']=='completed':
 folder=out/'fixtures/native0';release_facts=json.loads((folder/'constructor-result.json').read_text());before=json.loads((folder/'before-close.json').read_text())
 assert release_facts==r['rows'][0]['facts'] and r['constructor_gate_qualified']
 assert release_facts['schema']==p['schema'] and release_facts['resource_release']=='live-graph-release-v1'
 assert before=={k:release_facts[k] for k in before}
 assert before['specification']==p['expected_specification'] and before['muscles']==700
 assert before['preparation_steps']==200 and before['rollout_steps']==before['actors_constructed']==before['policy_inferences']==before['learner_updates']==0
 assert before['normalizer']=='none' and credit['used']==200 and not credit['pending']
 assert credit['settled']=={'native0:preparation':{'operations':1,'used':200}}
 worker=before['worker'];key=f"{worker['pid']}:{worker['start']}"
 native=[json.loads(line) for line in (folder/'traces/native-world-0.jsonl').read_text().splitlines()]
 closed=[json.loads(line) for line in (folder/'traces/worker-0.jsonl').read_text().splitlines()]
 assert len(native)==len(closed)==1 and native[0]['semantic']['phase']=='reset' and native[0]['semantic']['muscles']==700
 assert native[0]['owner']==closed[0]['owner']==worker and closed[0]['semantic']['phase']=='close'
 assert closed[0]['semantic']['state']==before['initial_state_hash']
 phases=[json.loads(line) for line in (folder/'resource-release.jsonl').read_text().splitlines()]
 assert phases==release_facts['release_phases']
 assert [e['phase'] for e in phases]==['before_close','inner_closed','graph_checked','released']
 assert all(e['identity']==worker for e in phases)
 assert phases[0]['state_hash']==phases[1]['state_hash']==before['initial_state_hash'] and phases[0]['muscles']==700
 assert phases[2]['weakrefs_dead']=={'env':True,'model':True,'data':True}
 assert all(a['monotonic_s']<=b['monotonic_s'] for a,b in zip(phases,phases[1:]))
 for e in [phases[0],phases[2]]:
  m=e['memory'];assert type(m['rss_bytes'])==int and m['rss_bytes']>0
  for z in [m['before'],m['after']]:
   assert z['pid']==worker['pid'] and z['start']==worker['start'] and z['state'] not in ('Z','X') and not z['flags']&4
 assert phases[2]['observed_rss_change_bytes']==phases[2]['memory']['rss_bytes']-phases[0]['memory']['rss_bytes']
 events=[json.loads(line) for line in (folder/'serial-close.jsonl').read_text().splitlines()]
 assert [(e['phase'],e.get('worker')) for e in events]==[('bound',None),('close_send',0),('joined',0),('closed',None)]
 assert events[2]['exitcode']==0 and events[2]['monotonic_s']-events[1]['monotonic_s']<=p['close_deadline_s']
 assert events[1]['identity']['pid']==events[2]['identity']['pid']==worker['pid']
 assert events[1]['monotonic_s']<=phases[0]['monotonic_s']<=closed[0]['monotonic_s']<=phases[1]['monotonic_s']<=phases[2]['monotonic_s']<=phases[3]['monotonic_s']<=events[2]['monotonic_s']
 watch=next(w for w in watchdogs if w['path'].endswith('native0-watchdog.json'))
 assert key in watch['retirement_records'] and watch['retirement_records'][key]['released']
 assert not list((folder/'guard-events').glob('*.jsonl')) and not list(folder.rglob('*.pt')) and not list(folder.rglob('*.zip'))
 assert len(watchdogs)==5


probe_audit=[]
for slot in p['slots'] if p['schema'] in ('joxing-process-exit-probe-v1','joxing-watch-publication-v1') else []:
 folder=out/'fixtures'/slot['id'];f=folder/'probe-result.json'
 for raw in [folder/'before-release.json',folder/'after-import.json']:
  if not raw.exists():continue
  proof=json.loads(raw.read_text())
  for key in ('worker_before','worker_after','root','server_before','server_after'):
   if key not in proof:continue
   z=proof[key];b=z['identity'];ids.append({'pid':b['pid'],'start_ticks':int(b['start'])})
   assert len(z['proc_evidence'])==4
   capture={}
   for name,d in z['proc_evidence'].items():
    file=Path(d['path']);assert file.resolve().is_relative_to(root.resolve())
    assert file.stat().st_size==d['bytes'] and sha(file)==d['sha256'];capture[name]=file.read_text()
   stats=[capture[k].rsplit(')',1)[1].split() for k in ('stat-before.txt','stat-after.txt')]
   assert all(int(capture[k].split('(',1)[0])==b['pid'] for k in ('stat-before.txt','stat-after.txt'))
   assert all(s[19]==b['start'] and s[0] not in ('Z','X') and not int(s[6])&4 for s in stats)
   assert z['parent_pid']==int(stats[1][1]) and z['after_threads']==int(stats[1][17])==1
   assert z['before_state']==stats[0][0] and z['after_state']==stats[1][0]
   assert z['before_flags']==int(stats[0][6]) and z['after_flags']==int(stats[1][6])
   status={k:v.strip() for line in capture['status.txt'].splitlines() if ':' in line for k,v in [line.split(':',1)]}
   assert int(status['Threads'])==z['threads']==1
   maps=capture['maps.txt'];assert hashlib.sha256(maps.encode()).hexdigest()==z['maps_sha256']
   assert z['heavy_mappings']==[s for s in ('libtorch','numpy','mujoco','openblas','mkl') if s in maps.lower()]
   if key!='worker_after':assert not z['heavy_mappings']
  for b in proof['auxiliaries']:ids.append({'pid':b['pid'],'start_ticks':int(b['start'])})
 if not f.exists():
  probe_audit.append({'id':slot['id'],'formal_status':next(s['status'] for s in r['rows'] if s['id']==slot['id']),'result_present':False});continue
 facts=json.loads(f.read_text());before=json.loads((folder/'before-release.json').read_text());after=json.loads((folder/'after-import.json').read_text())
 assert facts['method']==slot['method'] and facts['seed']==slot['seed']
 assert before['context']==before['ipc_context']==after['context']==slot['method']
 assert facts['worker']==before['worker_before']['identity']==after['worker_after']['identity']
 rootid=before['root']['identity'];wp=before['worker_before']['parent_pid']
 if slot['method']=='forkserver':
  server=before['server_before'];assert server['parent_pid']==rootid['pid'] and wp==server['identity']['pid']
  assert after['server_after']['identity']==server['identity']
 else:assert wp==rootid['pid'] and 'server_before' not in before
 payload=json.loads((folder/'payload.json').read_text());encoded=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
 digest=hashlib.sha256(encoded).hexdigest();assert digest==facts['payload_sha256']
 assert payload['versions']==p['runtime_versions'] and payload['seed']==slot['seed']
 assert payload['array_dtype']=='float64' and payload['tensor_dtype']=='torch.float64' and payload['shape']==[8] and payload['requires_grad'] is False
 import struct,math
 for k in ('array_hex','tensor_hex'):
  raw=bytes.fromhex(payload[k]);assert len(raw)==64
  values=struct.unpack('<8d',raw);assert all(math.isfinite(v) and i<=v<i+1 for i,v in enumerate(values))
 assert all(payload[k]==facts[k]==0 for k in ('physics','actors','updates','inference'))
 events=[json.loads(line) for line in (folder/'worker-events.jsonl').read_text().splitlines()]
 expected=['waiting_release','import_begin','ready','close_received','before_return','managed_finalizer']+(['python_atexit'] if slot['method']=='spawn' else [])
 assert [e['phase'] for e in events]==expected and all(e['identity']==facts['worker'] for e in events)
 assert events[0]['method']==slot['method'] and events[0]['seed']==slot['seed'] and events[2]['payload_sha256']==digest
 assert all(a['monotonic_s']<=b['monotonic_s'] for a,b in zip(events,events[1:]))
 assert events[4]['physics']==events[4]['updates']==0
 close=[json.loads(line) for line in (folder/'serial-close.jsonl').read_text().splitlines()]
 assert [(e['phase'],e.get('worker')) for e in close]==[('bound',None),('close_send',0),('joined',0),('closed',None)]
 duration=close[2]['monotonic_s']-close[1]['monotonic_s'];assert 0<duration<=2 and duration==facts['close_to_join_s'] and close[2]['exitcode']==0
 assert events[2]['monotonic_s']<=close[1]['monotonic_s']<=events[3]['monotonic_s']<=events[-1]['monotonic_s']<=close[2]['monotonic_s']
 watch=next(w for w in watchdogs if w['path'].endswith(slot['id']+'-watchdog.json'))
 watchkeys={(z['pid'],z['start_ticks']) for z in watch['owned_identities']}
 assert (facts['worker']['pid'],int(facts['worker']['start'])) in watchkeys
 assert all((z['pid'],int(z['start'])) in watchkeys for z in facts['auxiliaries'])
 probe_audit.append({'id':slot['id'],'formal_status':next(s['status'] for s in r['rows'] if s['id']==slot['id']),'method':slot['method'],'seed':slot['seed'],'payload_sha256':digest,'close_to_join_s':duration,'phases':expected,'root':rootid,'worker':facts['worker'],'auxiliaries':facts['auxiliaries'],'raw_proc_reconstructed':True,'retirement_observation':'observed' if watch.get('retirement_records') else 'not_observed'})
if p['schema']=='joxing-process-exit-probe-v1' and r['status']=='completed':
 assert len(probe_audit)==6 and all(s['formal_status']=='completed' for s in probe_audit)
 assert [s['id'] for s in probe_audit]==['spawn1097','forkserver1097','forkserver1098','spawn1098','spawn1099','forkserver1099']
 assert credit is None and not list(out.rglob('*.pt')) and not list(out.rglob('*.zip'))
 import statistics
 pairs=[]
 for seed in (1097,1098,1099):
  pair={s['method']:s for s in probe_audit if s['seed']==seed};assert pair['spawn']['payload_sha256']==pair['forkserver']['payload_sha256']
  pairs.append({'seed':seed,'spawn_s':pair['spawn']['close_to_join_s'],'forkserver_s':pair['forkserver']['close_to_join_s']})
 a=statistics.median(s['spawn_s'] for s in pairs);b=statistics.median(s['forkserver_s'] for s in pairs)
 verdict='supported_import_only' if b<=.5*a and all(s['forkserver_s']<=s['spawn_s'] for s in pairs) else 'not_supported_import_only'
 assert r['comparison']['pairs']==pairs and r['comparison']['spawn_median_s']==a and r['comparison']['forkserver_median_s']==b and r['comparison']['scientific_verdict']==verdict
 assert len(watchdogs)==(12 if p.get('watchdog_output_backend')=='private-tmpfs-v1' else 10)


publication_audit=[]
if p.get('watchdog_output_backend')=='private-tmpfs-v1':
 for f in [*[out/(slot['id']+'-watchdog-publication.json') for slot in p['slots']], *[x.parent/'watchdog-publication.json' for x in out.rglob('publication-control.json') if json.loads(x.read_text())['backend']=='private-tmpfs-v1']]:
  if not f.exists():continue
  c=json.loads(f.read_text());ids.append({'pid':c['owner']['pid'],'start_ticks':int(c['owner']['start'])})
  if c['status']=='copied_and_removed':
   assert not Path(c['directory']).exists() and Path(c['directory']).parent==Path('/dev/shm') and Path(c['directory']).name.startswith('joxing-watch-')
   assert c['allocation_stat']=={'uid':__import__('os').getuid(),'mode':448,'directory':True}
   assert ' - tmpfs ' in c['mount']['mountinfo'] and c['mount']['base']=='/dev/shm'
   assert 0<c['peak_ipc_bytes']<=c['cap_bytes']<=p['ipc_bytes']
   assert not c['task_live'] and not c['guard_live'] and c['copy_end_s']>=c['copy_begin_s']
   for d in c['copies']:
    target=Path(d['target']);assert target.resolve().is_relative_to(root.resolve())
    assert target.stat().st_size==d['bytes'] and sha(target)==d['sha256']
   w=json.loads(Path(c['canonical_report']).read_text());assert c['copy_begin_s']>=w['last_sample_monotonic_s']
   assert w['phase_timing']['self_final_write_included'] is False
   assert w['phase_timing']['max_wall_s']['probe']==w['max_probe_s']
   for e in w['phase_timing']['recent']:
    assert e['end_s']>=e['begin_s'] and e['wall_s']==e['end_s']-e['begin_s'] and e['cpu_s']>=0
   if f.parent!=out:assert c['copy_end_s']-c['copy_begin_s']>=.15
  elif r['status']=='completed':raise AssertionError('Incomplete private publication')
  publication_audit.append({'path':str(f.relative_to(root)),**c})
 if r['status']=='completed':
  assert len(publication_audit)==len(p['slots'])+1 and len(watchdogs)==len(p['slots'])+6
  if p['schema']=='joxing-watch-publication-v1':
   assert r['publication_smoke_qualified'] and len(probe_audit)==1 and probe_audit[0]['formal_status']=='completed' and probe_audit[0]['id']=='spawn1097'
  controls=[json.loads(f.read_text()) for f in out.rglob('publication-control.json')]
  assert len(controls)==2 and {c['backend'] for c in controls}=={'disk','private-tmpfs-v1'} and all(c['physics']==0 for c in controls)
  assert credit is None and not list(out.rglob('*.pt')) and not list(out.rglob('*.zip'))

slot_evidence=[]
for slot in p['slots']:
 folder=out/'fixtures'/slot['id'];evidence={'id':slot['id'],'registered_status':next(x['status'] for x in r['rows'] if x['id']==slot['id'])}
 f=folder/'trial/result.json'
 if f.exists():
  result=json.loads(f.read_text());evidence.update(actual_transitions=result['training_transitions'],actual_updates=result['updates'],initial_actor_hash=result['initial_actor_hash'],final_actor_hash=result['final_actor_hash'],recorded_training=result['counts'],recorded_evaluation_physics=result['evaluation_physics_substeps'],recorded_evaluation_preparation=result['evaluation_preparation_substeps'])
 evidence['trace_rows']={f.stem:len(f.read_text().splitlines()) for f in (folder/'traces').glob('*.jsonl')}
 slot_evidence.append(evidence)
ids={(i['pid'],i['start_ticks']):i for i in ids}
for i in ids.values():
 f=Path(f"/proc/{i['pid']}/stat")
 if f.exists():assert int(f.read_text().rsplit(')',1)[1].split()[19])!=i['start_ticks'],i
live=[]
for d in Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:
  if str((d/'cwd').resolve()).startswith(str(root)) and any(v in (d/'cmdline').read_bytes() for v in (b'python',b'/uv ')):live.append(int(d.name))
 except OSError:pass
assert not live
files={str(f.relative_to(root)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted((root/'runs/v0.0.11').rglob('*')) if f.is_file()}
scratch=sum(d['bytes'] for n,d in files.items() if '/fixtures/' in n);receipts=sum(d['bytes'] for n,d in files.items() if '/fixtures/' not in n)
assert scratch<=p['fixture_bytes'] and receipts<=p['receipt_bytes']
print(json.dumps({'schema':'joxing-deferred-native-independent-v1','status':'verified_'+r['status']+'_attempt','source_commit':r['source_commit'],'protocol_sha256':r['protocol_sha256'],'source_files_verified':len(p['source_files']),'historical_objects_verified':len(p['required_git_objects']),'complete_superproject_history':True,'case_rows':cases,'watchdog_reports':watchdogs,'credit_audit':credit,'credit_events_verified':len(credit_events),'slot_evidence':slot_evidence,'constructor_facts':constructor_facts,'serial_facts':serial_facts,'release_facts':release_facts,'probe_audit':probe_audit,'publication_audit':publication_audit,'registered_cases':len(p['test_node_ids']),'counts':{s:sum(c['status']==s for c in cases) for s in ('passed','failed','error','xfail','skipped')},'exited_recorded_identities':list(ids.values()),'live_writers':live,'raw_manifest':files,'fixture_bytes':scratch,'receipt_bytes':receipts,'supervision':r,'observed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}))
"""
remote=remote.replace('PAYLOAD',repr(base64.b64encode(json.dumps(p).encode()).decode()))
r=subprocess.run(['ssh','-o','ConnectTimeout=15','xbabe1','python3 -'],input=remote,text=True,capture_output=True,timeout=90)
if r.returncode:print(r.stderr);r.check_returncode()
a=json.loads(r.stdout);Path('artifacts/v0.0.11/development/'+label+'-integrity.json').write_text(json.dumps(a,indent=2)+'\n')
raw=subprocess.check_output(['ssh','-o','ConnectTimeout=15','xbabe1','tar -C '+p['deployment_root']+' -cf - -- '+' '.join(a['raw_manifest'])],timeout=90)
out=Path('runs/v0.0.11/'+label+'-evidence');out.mkdir(exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(raw)) as t:
 assert set(t.getnames())==set(a['raw_manifest']);t.extractall(out,filter='data')
for n,d in a['raw_manifest'].items():
 f=out/n;assert f.stat().st_size==d['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==d['sha256']
dest=Path('artifacts/v0.0.11/'+label);dest.mkdir(exist_ok=False)
for f in (out/p['output']).iterdir():
 if f.is_file():shutil.copyfile(f,dest/f.name)
(dest/'raw-manifest.json').write_text(json.dumps(a['raw_manifest'],indent=2)+'\n')
transfer={'status':'passed','raw_files':len(a['raw_manifest']),'raw_bytes':sum(d['bytes'] for d in a['raw_manifest'].values()),'tar_sha256':hashlib.sha256(raw).hexdigest(),'all_raw_names_sizes_hashes_verified':True}
Path('artifacts/v0.0.11/development/'+label+'-transfer.json').write_text(json.dumps(transfer,indent=2)+'\n')
print(json.dumps({k:a[k] for k in ('status','source_files_verified','registered_cases','counts','fixture_bytes','receipt_bytes')}));print(json.dumps({'exited':len(a['exited_recorded_identities']),**transfer}))
