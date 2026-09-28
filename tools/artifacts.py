"""Restore immutable release data and resolve hash-checked upstream meshes."""
from pathlib import Path
import argparse,hashlib,importlib.util,io,json,shutil,urllib.request,zipfile

ROOT=Path(__file__).resolve().parents[1]
def digest(data):return hashlib.sha256(data).hexdigest()

def verify():
    count=0
    for line in (ROOT/'release-data/SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1)
        if digest((ROOT/name).read_bytes())!=expected:raise ValueError(f'Changed release file: {name}')
        count+=1
    print(f'Verified {count} immutable release files')
    return count

def restore():
    verify()
    for source in sorted((ROOT/'release-data').rglob('*')):
        if not source.is_file() or len(source.relative_to(ROOT/'release-data').parts)<2:continue
        destination=ROOT/'artifacts'/source.relative_to(ROOT/'release-data')
        if destination.exists():
            if digest(source.read_bytes())!=digest(destination.read_bytes()):
                raise FileExistsError(f'Preserving changed experiment file: {destination}')
            continue
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    print('Restored release data to artifacts/ without overwriting changes')

def fetch_assets(mjlab_root=None):
    manifest=json.loads((ROOT/'release-data/ASSETS.json').read_text())
    files=manifest['files'];missing=[]
    for name,item in files.items():
        path=ROOT/'artifacts'/name
        if path.exists():
            if digest(path.read_bytes())!=item['sha256']:raise ValueError(f'Changed mesh: {path}')
        else:missing.append((name,item))
    if not missing:print('All upstream meshes already verified');return
    package=Path(mjlab_root) if mjlab_root else None
    if package is None:
        spec=importlib.util.find_spec('mjlab')
        if spec and spec.origin:package=Path(spec.origin).parent
    wheel=None
    if package is None:
        with urllib.request.urlopen('https://pypi.org/pypi/mjlab/1.2.0/json',timeout=60) as response:
            metadata=json.load(response)
        entry=next(e for e in metadata['urls'] if e['filename']=='mjlab-1.2.0-py3-none-any.whl')
        if entry['digests']['sha256']!=manifest['wheel_sha256']:raise ValueError('Unexpected upstream wheel hash')
        print('Fetching pinned mjlab wheel for mesh assets only; not installing mjlab')
        with urllib.request.urlopen(entry['url'],timeout=120) as response:data=response.read()
        if digest(data)!=manifest['wheel_sha256']:raise ValueError('Corrupt upstream wheel')
        wheel=zipfile.ZipFile(io.BytesIO(data))
    for name,item in missing:
        data=(package/item['mjlab_path']).read_bytes() if package else wheel.read('mjlab/'+item['mjlab_path'])
        if digest(data)!=item['sha256']:raise ValueError(f'Upstream mesh mismatch: {name}; use mjlab 1.2.0')
        path=ROOT/'artifacts'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    print(f'Installed and verified {len(missing)} mesh references')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','restore','assets']);p.add_argument('--mjlab-root');a=p.parse_args()
    if a.action=='verify':verify()
    elif a.action=='restore':restore()
    else:fetch_assets(a.mjlab_root)
