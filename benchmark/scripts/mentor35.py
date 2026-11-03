"""Explicitly authorized SD1.4/r4 manipulation-transfer subset; native per-image metrics."""
import argparse,os,sys,json,subprocess,sqlite3,time,gc
from pathlib import Path
from benchmark.shared.process_title import configure
from benchmark.shared.data.protocol import validate,digest
from benchmark.shared.evaluation.native_metrics import evaluate,calibrate
PROJECT=Path(__file__).resolve().parents[2]
METHODS=['stable_signature','gaussmarker','trustmark','grow','serum']
ATTACKS=['flip','rotation15','crop75','drop96','gaussian005','brightness07','blur15']

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
def db(root):return sqlite3.connect(root/'queue.sqlite',timeout=60)

def init(root,assets,calibration):
    prompts=read(PROJECT/'benchmark/shared/data/prompts.json');validate(prompts,read(calibration))
    root.mkdir(parents=True,exist_ok=True)
    config={'assets':str(assets.resolve()),'calibration':str(calibration.resolve()),'rank':4,'steps':20000,'batch':4,
        'prompt_sha256':digest(PROJECT/'benchmark/shared/data/prompts.json'),'calibration_sha256':digest(calibration),
        'student_count':35,'aggregate':False,'process_title':'VLLM::Enginecore',
        'teacher_training_membership':'unverified for public pretrained methods; training-free GROW',
        'student_test':'unmanipulated outputs; manipulation applies ONLY to training targets',
        'thresholds':'native per-image, clean-teacher-only; clean threshold for unmanipulated student outputs'}
    if (root/'config.json').exists() and read(root/'config.json')!=config:raise ValueError('Changed experiment identity')
    write(root/'config.json',config);c=db(root)
    c.execute('create table if not exists jobs(id text primary key,kind text,method text,attack text,state text,pid integer,error text)')
    for m in METHODS:
        c.execute('insert or ignore into jobs values(?,?,?,?,?,?,?)',(m,'teacher',m,'','pending',0,''))
        for attack in ATTACKS:c.execute('insert or ignore into jobs values(?,?,?,?,?,?,?)',(m+'/'+attack,'student',m,attack,'pending',0,''))
    c.commit();c.close()

def setup(root,method):
    config=read(root/'config.json');assets=read(config['assets']);workspace=root/'teachers'/method
    os.environ['BASELINE_WORK']=str(workspace)
    import common
    common.ROOT=workspace
    # Read historical format defaults from the deployed repo, not another run.
    cfg={**read(PROJECT/'config/default.json'),**assets};cfg.update(inference_steps=50,cfg=7.5)
    write(workspace/'config.json',cfg)
    from benchmark.methods.native import create
    return create(method,workspace)

def coordinates(model,folder,attack):
    import torch
    from common import pixels
    from benchmark.shared.manipulations.saturday import transform
    rows=[];payload=[]
    for i,f in enumerate(sorted(Path(folder).glob('sample_*.png'))):
        x=pixels([f])
        if attack!='clean':x=transform(x[0],attack,i)[None]
        rows.extend(model.coordinates(x,[i]).tolist())
        if hasattr(model,'last_bits'):payload.extend(model.last_bits.tolist())
    return rows,payload or None

def teacher(root,method):
    import torch,numpy as np,hashlib
    from common import save_images,manifest
    config=read(root/'config.json');p=root/'teachers'/method;model=setup(root,method)
    prompts=read(PROJECT/'benchmark/shared/data/prompts.json')
    splits={k:prompts[k] for k in ['train','validation','test']}
    if method in ['gaussmarker','serum']:splits['calibration']=read(config['calibration'])
    for split,rows in splits.items():
        folder=p/'images'/('teacher_wm_'+split if split!='calibration' else 'clean_calibration');folder.mkdir(parents=True,exist_ok=True)
        batch=1 if method in ['gaussmarker','grow'] else 8
        for start in range(0,len(rows),batch):
            ids=list(range(start,min(start+batch,len(rows))))
            if all((folder/f'sample_{i:05d}.png').exists() for i in ids):continue
            texts=[rows[i]['text'] for i in ids];seeds=[int(rows[i]['image_id']) for i in ids]
            if split=='calibration':
                with torch.inference_mode():images=model.pipe(texts,height=512,width=512,num_inference_steps=50,guidance_scale=7.5,generator=[torch.Generator('cuda').manual_seed(s) for s in seeds]).images
            else:images=model.generate(texts,seeds)
            save_images(images,folder,ids);write(p/'status.json',{'stage':'generate','split':split,'images':start+len(ids),'pid':os.getpid()})
        manifest(folder,rows)
    write(p/'prompts.json',prompts)
    from PIL import Image
    lookup={}
    for i in range(5000):
        x=np.asarray(Image.open(p/'images/teacher_wm_train'/f'sample_{i:05d}.png').convert('RGB')).transpose(2,0,1).copy()
        lookup[hashlib.sha256(x.tobytes()).hexdigest()]=i
    write(p/'pixel_index.json',lookup)
    thresholds={}
    for attack in ['clean',*ATTACKS]:
        if method in ['gaussmarker','serum']:
            score,_=coordinates(model,p/'images/clean_calibration',attack)
            thresholds[attack]=calibrate([v[0] for v in score],method=method,clean_manifest_sha256=digest(p/'images/clean_calibration/image_hashes.json'),output=p/'thresholds'/f'{attack}.json',attack=attack)
        values,bits=coordinates(model,p/'images/teacher_wm_test',attack)
        result=evaluate(method,values,getattr(model,'key',None),thresholds.get(attack),bits)
        write(p/'results'/f'{attack}.json',{'method':method,'model':'teacher','attack':attack,**result})
    write(p/'complete.json',{'method':method,'prompts_sha256':config['prompt_sha256'],'aggregate':False})

def student(root,method,attack):
    from benchmark.shared.student_training.integration import sd14
    import torch
    config=read(root/'config.json');assets=read(config['assets']);p=root/'students'/method/attack
    source=root/'teachers'/method;model_path=assets['student_path']
    if not (p/'trained.json').exists():
        v=sd14(p,model_path,source/'images/teacher_wm_train',4)
        target=p/'images/teacher_wm_train';target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.symlink_to(source/'images/teacher_wm_train',target_is_directory=True)
        driver=v['helper']/'driver.py';text=driver.read_text()
        text=text.replace("    ns['main']()","    from benchmark.shared.student_training.training_channel import install\n    install()\n    ns['main']()")
        driver.write_text(text)
        cmd=v['builder'](v['config'],v['assets'],p,v['branch']);cmd[cmd.index('--checkpointing_steps')+1]='1250'
        checkpoints=list((p/'students/sd14_r4').glob('checkpoint-*'))
        if checkpoints:cmd+=['--resume_from_checkpoint',str(max(checkpoints,key=lambda x:int(x.name.split('-')[-1])))]
        env=os.environ.copy();env.update(BASELINE_TRAIN_ATTACK=attack,BASELINE_PIXEL_INDEX=str(source/'pixel_index.json'))
        subprocess.run([sys.executable,str(driver),'--trainer',cmd[1],'--identity-file',str(v['identity']),*cmd[2:]],env=env,check=True)
        write(p/'trained.json',{'steps':20000,'rank':4,'attack':attack})
    from benchmark.shared.evaluation.student_images import generate
    if not (p/'test_images/image_hashes.json').exists():generate('sd14','r4',model_path,p,read(source/'prompts.json')['test'],p/'test_images')
    gc.collect();torch.cuda.empty_cache();model=setup(root,method)
    values,bits=coordinates(model,p/'test_images','clean')
    threshold=read(source/'thresholds/clean.json') if method in ['gaussmarker','serum'] else None
    result=evaluate(method,values,getattr(model,'key',None),threshold,bits)
    write(p/'results.json',{'method':method,'model':'sd14_r4','training_attack':attack,'test_attack':'clean','aggregate':False,**result})

def worker(root):
    while True:
        c=db(root);c.execute('begin immediate');chosen=None
        for row in c.execute("select id,kind,method,attack from jobs where state='pending' order by kind desc,id").fetchall():
            if row[1]=='teacher' or (root/'teachers'/row[2]/'complete.json').exists():chosen=row;break
        if chosen:c.execute("update jobs set state='running',pid=? where id=?",(os.getpid(),chosen[0]))
        c.commit()
        if not chosen:
            remaining=c.execute("select count(*) from jobs where state in ('running','pending')").fetchone()[0];c.close()
            if not remaining:return
            time.sleep(15);continue
        c.close();ident,kind,method,attack=chosen;log=root/'logs'/(ident.replace('/','_')+'.log');log.parent.mkdir(exist_ok=True)
        with log.open('a') as out:r=subprocess.run([sys.executable,'-m','benchmark.scripts.mentor35','task','--root',str(root),'--method',method,'--kind',kind,'--attack',attack],stdout=out,stderr=subprocess.STDOUT)
        c=db(root);c.execute('update jobs set state=?,error=? where id=?',('completed' if r.returncode==0 else 'failed','' if r.returncode==0 else str(log),ident));c.commit();c.close()
        print('DONE' if r.returncode==0 else 'FAILED',kind,ident,flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=['init','launch','worker','task','status']);a.add_argument('--root',type=Path,required=True)
    a.add_argument('--assets',type=Path);a.add_argument('--calibration',type=Path);a.add_argument('--method');a.add_argument('--kind');a.add_argument('--attack',default='');args=a.parse_args();root=args.root.resolve()
    if args.action=='init':init(root,args.assets,args.calibration);print('PREPARED 35 students; no aggregate');return
    if args.action=='status':
        c=db(root);print(c.execute('select kind,state,count(*) from jobs group by kind,state').fetchall());c.close();return
    configure()
    if args.action=='worker':worker(root)
    elif args.action=='task':teacher(root,args.method) if args.kind=='teacher' else student(root,args.method,args.attack)
    else:
        if (root/'launch.json').exists():raise ValueError('Already launched; check status')
        jobs=[]
        for gpu in ['0','0','1','1']:
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TQDM_DISABLE='1',HF_HUB_DISABLE_PROGRESS_BARS='1')
            log=root/'logs'/f'worker_{len(jobs)}.log';log.parent.mkdir(exist_ok=True)
            with log.open('a') as out:p=subprocess.Popen([sys.executable,'-m','benchmark.scripts.mentor35','worker','--root',str(root)],cwd=PROJECT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            jobs.append({'pid':p.pid,'gpu':gpu})
        write(root/'launch.json',jobs);print('LAUNCHED',jobs)
if __name__=='__main__':main()
