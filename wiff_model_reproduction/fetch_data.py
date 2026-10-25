"""Download the paper's official training archive with MD5 verification."""
from pathlib import Path
import argparse,hashlib,json,urllib.request

ROOT=Path(__file__).resolve().parent
URL="https://zenodo.org/api/records/5108636/files/Training_converged.mat/content"
MD5="84547e2fcfa2821ccf9a2dd3057c3588"
SIZE=753361284


def digest(path,algorithm="md5"):
    value=hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'data')
    args=parser.parse_args();args.directory.mkdir(exist_ok=True,parents=True)
    target=args.directory/'Training_converged.mat'
    if target.exists() and target.stat().st_size==SIZE and digest(target)==MD5:
        print('Existing data checksum verified.');return
    temporary=target.with_suffix('.mat.part');total=0
    request=urllib.request.Request(URL,headers={'User-Agent':'WIFF-reproduction/1.0'})
    with urllib.request.urlopen(request,timeout=120) as response,temporary.open('wb') as out:
        for block in iter(lambda:response.read(1024*1024),b''):
            out.write(block);total+=len(block)
            if total%(64*1024*1024)==0:print(f'{total/1024**2:.0f} / {SIZE/1024**2:.0f} MiB',flush=True)
    if total!=SIZE or digest(temporary)!=MD5:raise RuntimeError('Data checksum or size mismatch')
    temporary.replace(target)
    print('Downloaded official data; MD5 verified.')


if __name__=='__main__':main()
