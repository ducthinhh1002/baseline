"""Pinned upstream sources, public teacher artifacts and clean student checkpoints."""
from pathlib import Path
import argparse,os,json,subprocess,time,urllib.request,zipfile,hashlib,concurrent.futures
from common import CACHE,ROOT,REPO,config,read,write,freeze,sha
MODELS={
 'teacher':('Manojb/stable-diffusion-2-1-base','0094d483a120f3f33dafbd187ea4aa60d10de75c'),
 'sd14':('CompVis/stable-diffusion-v1-4','133a221b8aa7292a167afc5127cb63fb5005638b'),
 'sana600m':('Efficient-Large-Model/Sana_600M_512px_diffusers','83d7a190bfd1fd070570a793d2dab5c7a3231b9d'),
 'pixart512':('PixArt-alpha/PixArt-XL-2-512x512','50f702106901db6d0f8b67eb88e814c56ded2692')}
SERUM_CHECKPOINT_SHA='e991a694342545fa4f59cc5b3c94b6e3a93bf831dd4d507207f52d1c167c05c4'
SERUM_CACHE_SHA='95877f6ab0305b92897c5f6fdba7fcd373b9031728f3e69599ffebfb6d2bc593'
def url_file(url,p,digest=None,md5=None):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        temp=p.with_suffix('.partial')
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url,timeout=120) as src,temp.open('wb') as dest:
                    while data:=src.read(4*1024*1024):dest.write(data)
                if digest and sha(temp)!=digest:raise ValueError('SHA256 mismatch: '+str(p))
                if md5 and hashlib.md5(temp.read_bytes()).hexdigest()!=md5:raise ValueError('Publisher MD5 mismatch: '+str(p))
                temp.replace(p);break
            except Exception:
                if attempt==3:raise
                time.sleep(3)
    if digest and sha(p)!=digest:raise ValueError('SHA256 mismatch: '+str(p))
    if md5 and hashlib.md5(p.read_bytes()).hexdigest()!=md5:raise ValueError('MD5 mismatch: '+str(p))
    freeze(p.with_suffix(p.suffix+'.source.json'),{'url':url,'sha256':sha(p),'bytes':p.stat().st_size})
    return str(p.resolve())
def sources():
    for name,entry in read(REPO/'config/upstream.json').items():
        p=CACHE/'upstream'/name
        if not p.exists():
            subprocess.run(['git','clone','--no-checkout','--filter=blob:none',entry['url'],str(p)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            subprocess.run(['git','-C',str(p),'checkout',entry['revision']],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        actual=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
        if actual!=entry['revision']:raise RuntimeError('Upstream source revision mismatch: '+name)
def hub(repo,revision,name):
    from huggingface_hub import snapshot_download
    c=config();override=c.get('model_overrides',{}).get(name)
    if override:
        p=Path(override).resolve();assert (p/'model_index.json').exists()
    else:p=Path(snapshot_download(repo,revision=revision,cache_dir=str(CACHE/'hub'),allow_patterns=['*.json','*.txt','*.model','*.tiktoken','*.safetensors'],ignore_patterns=['*.fp16*','*.onnx','*.bin']))
    files={str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file()}
    entry={'repo':repo,'revision':revision,'path':str(p),'files':files}
    freeze(CACHE/'models'/f'{name}.json',entry);return str(p)
def hf_file(repo,revision,file):
    from huggingface_hub import hf_hub_download
    p=Path(hf_hub_download(repo,file,revision=revision,cache_dir=str(CACHE/'hub')))
    freeze(CACHE/'hf_artifacts'/repo.replace('/','--')/(file+'.json'),{'repo':repo,'revision':revision,'file':file,'sha256':sha(p)})
    return str(p)
def serum():
    import gdown
    folder=CACHE/'teachers/serum';folder.mkdir(parents=True,exist_ok=True)
    for file,id,digest in [('checkpoint.pth','12NTXeyIMFX1NWYzCJDeRX54T7BgEvXuQ',SERUM_CHECKPOINT_SHA),('author_cache.zip','17MFAZ2rsmaIlRNghzTbShLE73t6QCCrN',SERUM_CACHE_SHA)]:
        p=folder/file
        if file=='author_cache.zip' and (folder/'backbone.json').exists():continue
        if not p.exists():
            tmp=p.with_suffix('.partial');gdown.download(id=id,output=str(tmp),quiet=True)
            if sha(tmp)!=digest:raise RuntimeError('Author SERUM artifact mismatch')
            tmp.replace(p)
        if sha(p)!=digest:raise RuntimeError('Author SERUM artifact mismatch')
    if not (folder/'backbone.json').exists():
        dest=folder/'author_cache';dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(folder/'author_cache.zip') as z:
            for f in z.namelist():
                target=(dest/f).resolve()
                if not(target==dest.resolve() or dest.resolve() in target.parents):raise ValueError('Unsafe zip path')
            z.extractall(dest)
        files=list(dest.rglob('snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06/model_index.json'))
        if len(files)!=1:raise RuntimeError('Author SD2.1 backbone snapshot missing')
        freeze(folder/'backbone.json',{'path':str(files[0].parent.resolve()),'archive_sha256':SERUM_CACHE_SHA})
def teacher_assets(method):
    dest=CACHE/'teachers'/method;dest.mkdir(parents=True,exist_ok=True)
    if method=='stable_signature':
        url_file('https://dl.fbaipublicfiles.com/ssl_watermarking/sd2_decoder.pth',dest/'sd2_decoder.pth','36f3a926ba080a0fe29952cd3895661736da68619a71c8166d5c44de3612965a')
        url_file('https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt',dest/'extractor.pt')
    elif method=='gaussmarker':
        files={n:hf_file('SunnierLeee/GaussMarker-pretrained','880fcb1e43064e2c0921fced3745f96e9a8fa2ae',n) for n in ['model_final.pth','w1_256.pth','w2.pth']}
        files['classifier']=str((CACHE/'upstream/GaussMarker/sd21_cls2.pkl').resolve())
        freeze(dest/'files.json',files)
    elif method=='trustmark':
        if config()['trustmark_model_type']!='Q':raise ValueError('This pinned baseline config uses official TrustMark-Q')
        models=CACHE/'upstream/trustmark/python/trustmark/models'
        hashes={'trustmark_Q.yaml':'fe40df84a7feeebfceb7a7678d7e6ec6','decoder_Q.ckpt':'4ced90e9cfe13e3295ad082887fe9187','encoder_Q.ckpt':'700328b8754db934b2f6cb5e5185d81f'}
        for name,digest in hashes.items():url_file('https://cai-watermark.adobe.net/watermarking/trustmark-models/'+name,models/name,md5=digest)
    elif method=='serum':serum()
    elif method!='grow':raise ValueError(method)
    freeze(dest/'download_complete.json',{'method':method,'upstream_revision':read(REPO/'config/upstream.json')[{'gaussmarker':'GaussMarker','grow':'GROW','serum':'SERUM','trustmark':'trustmark','stable_signature':'stable_signature'}[method]]['revision']})
def main(bases_only=False):
    sources()
    for name in ['teacher',*config()['students']]:
        repo,revision=MODELS[name];hub(repo,revision,name)
    if bases_only:return
    failures={}
    for method in config()['methods']:
        try:teacher_assets(method);print('DOWNLOADED teacher',method,flush=True)
        except Exception as e:failures[method]=repr(e)
    write(ROOT/'download_failures.json',failures)
    if failures:raise RuntimeError('Some native author artifacts could not be downloaded: '+str(failures))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bases-only',action='store_true');a=p.parse_args();main(a.bases_only)
