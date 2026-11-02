"""User-invoked GPU entry point; import/plan do not load torch or download anything."""
import argparse,os,sys,json,time,random,shutil,gc,hashlib
from pathlib import Path
from .integration import dit_source,sha

def pixels(paths):
    import torch,numpy as np
    from PIL import Image
    return torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).permute(2,0,1).float()/127.5-1 for p in paths]).cuda()

def numerical_policy():
    import torch
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--arch',choices=['sana600m','pixart512'],required=True)
    p.add_argument('--mode',choices=['full','r4','r16','r32'],required=True);a=p.parse_args()
    import numpy as np,fcntl
    rank=None if a.mode=='full' else int(a.mode[1:]);job=a.job.resolve();job.mkdir(parents=True,exist_ok=True)
    def read_json(p):return json.loads(Path(p).read_text())
    def write_json(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2))
    def freeze(p,v):
        if Path(p).exists() and read_json(p)!=v:raise ValueError('Frozen identity changed: '+str(p))
        write_json(p,v)
    identity={'arch':a.arch,'rank':rank,'model':str(a.model.resolve()),'prompts':sha(a.source/'prompts.json'),
        'images':sha(a.source/'images/teacher_wm_train/image_hashes.json'),'reused_training_source':sha(Path(__file__).parent/'reference/cross_arch_jobs.py'),
        'patch':'path/owner-identity replacement; rank or full trainable scope; no watermark loss',
        'full_sana_new_configuration':a.arch=='sana600m' and rank is None}
    freeze(job/'model_identity.json',{'path':str(a.model.resolve())});freeze(job/'experiment_identity.json',identity)
    def status(p,stage,**kw):write_json(p/'status.json',{'stage':stage,**kw})
    namespace=dict(ROOT=job,SOURCE=a.source.resolve(),paths=lambda name:job,identity=lambda name:identity,
        status=status,read_json=read_json,write_json=write_json,freeze=freeze,sha256=sha,Path=Path,
        os=os,sys=sys,json=json,time=time,random=random,shutil=shutil,fcntl=fcntl,gc=gc,hashlib=hashlib)
    code=dit_source(rank);freeze(job/'integration.json',{'generated_source_sha256':hashlib.sha256(code.encode()).hexdigest()})
    exec(compile(code,str(job/'reused_student.py'),'exec'),namespace)
    namespace['cache'](a.arch);namespace['train'](a.arch)
    print('DONE student',a.arch,a.mode,flush=True)

if __name__=='__main__':main()
