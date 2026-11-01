#!/usr/bin/env python3
"""Acquire the provider's four benchmark inputs and check recorded fingerprints. This script is not run by preflight."""
from pathlib import Path
import argparse,csv,hashlib,json,shutil,sys,urllib.request,zipfile,gzip
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['ml1m','beauty','sports','toys'],required=True);p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();workspace=a.workspace.resolve()
 providers=json.loads((ROOT/'data_manifest/providers.json').read_text(encoding='utf-8'));names={'ml1m':'MovieLens-1M','beauty':'Beauty','sports':'Sports','toys':'Toys'};source=next(s for s in providers['sources']if s['dataset']==names[a.dataset]);expected={r['path']:r['sha256']for r in csv.DictReader((ROOT/'data_manifest/expected_sha256.csv').open(encoding='utf-8-sig',newline=''))}
 download=(workspace/'downloads'/source['download'].rsplit('/',1)[-1]).with_suffix(Path(source['download']).suffix+'.downloaded');download.parent.mkdir(parents=True,exist_ok=True)
 if download.exists():raise SystemExit('Download staging file already exists; inspect it before retrying: '+str(download))
 request=urllib.request.Request(source['download'],headers={'User-Agent':'KAIS-research-reproduction/1.0'})
 with urllib.request.urlopen(request,timeout=120)as inp,download.open('xb')as out:shutil.copyfileobj(inp,out)
 verified=[]
 if source['kind']=='zip':
  with zipfile.ZipFile(download)as z:
   assert z.testzip()is None
   staged=[]
   for relative in source['verify_members']:
    member=relative.removeprefix(source['restore_directory']+'/');data=z.read(member);digest=hashlib.sha256(data).hexdigest();verified.append({'path':relative,'sha256':digest,'matches_recorded_input':digest==expected[relative]});staged.append((relative,data))
   if not all(x['matches_recorded_input']for x in verified):print(json.dumps({'status':'DOWNLOADED_INPUT_HASH_MISMATCH','download':str(download),'files':verified},indent=2));return 3
   for relative,data in staged:
    dest=(workspace/relative).resolve();assert dest.is_relative_to(workspace)
    if dest.exists():assert sha(dest)==expected[relative],'Different existing file: '+str(dest)
    else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
   readme=workspace/source['restore_directory']/'ml-1m/README'
   if not readme.exists():readme.write_bytes(z.read('ml-1m/README'))
 else:
  relative=source['restore_path'];digest=sha(download);verified.append({'path':relative,'sha256':digest,'matches_recorded_input':digest==expected[relative]})
  if digest!=expected[relative]:
   print(json.dumps({'status':'DOWNLOADED_CONTAINER_HASH_MISMATCH','download':str(download),'files':verified,'note':'Kept the downloaded file for inspection. A compressed-byte mismatch is not a determination about decompressed interactions. Do not silently substitute the historical expected hash.'},indent=2));return 3
  if source.get('expected_review_rows'):
   count=0
   with gzip.open(download,'rb')as stream:
    for line in stream:
     if line.strip():count+=1
   assert count==source['expected_review_rows'],('Unexpected complete review-row count',count)
   verified[-1]['complete_review_rows']=count;verified[-1]['gzip_eof_crc_passed']=True
   assert download.stat().st_size==source['expected_bytes']
  dest=(workspace/relative).resolve();assert dest.is_relative_to(workspace)
  if dest.exists():assert sha(dest)==digest,'Different existing file: '+str(dest)
  else:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(download,dest)
 print(json.dumps({'status':'RAW_INPUT_ACQUIRED_AND_HASH_VERIFIED','provider':source['landing_page'],'files':verified,'training_executed':False},indent=2));return 0
if __name__=='__main__':sys.exit(main())
