"""Independent stdlib/NumPy decoding; zero physics, no production storage reader."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/ubuntu/joxing/v011-storage-qualification-20261027')
OUT = ROOT / 'runs/v0.0.11/storage-qualification-01'
SUP = ROOT / 'runs/v0.0.11/storage-supervision-01'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def describe(array):
    return {'dtype': array.dtype.str, 'shape': list(array.shape), 'bytes': array.nbytes,
            'sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}


packet = json.loads((ROOT / 'experiments/v0.0.11/storage-qualification-protocol.json').read_text())
result = json.loads((OUT / 'result.json').read_text())
supervision = json.loads((SUP / 'result.json').read_text())
assert result['status'] == 'passed' and supervision['status'] == 'completed'
assert result['new_physics'] == 0 and len(result['rows']) == 112
assert all(r['status'] == 'passed' for r in result['rows'])
for name, expected in {**packet['source_files'], **packet['upstream_files']}.items():
    path = ROOT / name
    assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], name
rows = []
for corpus in packet['corpora']:
    original = Path(corpus['root']); packed = OUT / 'packed' / corpus['id']
    assert {str(f.relative_to(original)) for f in original.rglob('*') if f.is_file()} == set(corpus['files'])
    for name, expected in corpus['files'].items():
        source = original / name; target = packed / name
        assert source.stat().st_size == expected['bytes'] and sha(source) == expected['sha256'], name
        if source.suffix in ('.json', '.npz'):
            manifest = json.loads((target.parent / 'storage-manifest.json').read_text())
            entry = manifest['files'][target.name]
            stored = target.parent / entry['stored']
            assert stored.stat().st_size == entry['bytes'] and sha(stored) == entry['sha256']
            if source.suffix == '.json':
                with gzip.open(stored, 'rt') as stream:
                    actual = json.load(stream)
                baseline = json.loads(source.read_text())
                assert actual == baseline
                raw = canonical(baseline)
                assert entry['decoded'] == {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            else:
                with np.load(source, allow_pickle=False) as a, np.load(stored, allow_pickle=False) as b:
                    assert set(a.files) == set(b.files)
                    descriptions = {}
                    for key in a.files:
                        aa, bb = a[key], b[key]
                        assert aa.dtype == bb.dtype and aa.shape == bb.shape
                        assert aa.tobytes(order='C') == bb.tobytes(order='C')
                        descriptions[key] = describe(aa)
                    assert descriptions == entry['decoded']
        else:
            assert sha(target) == expected['sha256']
        rows.append({'corpus': corpus['id'], 'path': name, 'status': 'passed'})
identities = [supervision['process']]
for stage in supervision['stages']:
    identities += [stage['process']] + stage.get('owned_processes', [])
live = []
for row in identities:
    path = Path('/proc') / str(row['pid']) / 'stat'
    try:
        stat = path.read_text().split(') ')[1].split()
        if int(stat[19]) == row['start_ticks'] and stat[0] != 'Z':
            live.append(row)
    except FileNotFoundError:
        pass
assert not live, live
unknown = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        cwd = str((proc / 'cwd').resolve()); cmd = (proc / 'cmdline').read_bytes()
        if '/joxing/' in cwd and (b'python' in cmd or b'/uv ' in cmd):
            unknown.append({'pid': int(proc.name), 'cwd': cwd})
    except OSError:
        pass
assert not unknown, unknown
files = {str(p.relative_to(ROOT / 'runs/v0.0.11')): {'bytes': p.stat().st_size, 'sha256': sha(p)}
         for folder in [OUT, SUP] for p in folder.rglob('*') if p.is_file()}
print(json.dumps({'schema': 'joxing-independent-storage-audit-v1', 'status': 'passed',
                  'source_sha': result['source_sha'], 'new_physics': 0, 'rows': rows,
                  'verified_source_upstream_files': len(packet['source_files']) + len(packet['upstream_files']),
                  'owned_identities': identities, 'live_recorded': live, 'unknown_owned': unknown,
                  'files': files, 'retained_bytes': sum(x['bytes'] for x in files.values()),
                  'qualification': {k: v for k, v in result.items() if k != 'rows'}}))
