"""Teacher robustness + 105 ordinary rank4 students; quiet, resumable process queue."""
from pathlib import Path
import argparse,os,sys,json,time,subprocess,sqlite3,traceback,contextlib,gc
from common import ROOT,CACHE,REPO,read,write,freeze,sha,config,status,pixels,image_name,manifest,save_images,METHODS,STUDENTS,code_identity

def initialize(args):
    c=read(args.config)
    if args.prompts:c['prompt_source']=str(Path(args.prompts).resolve())
    if args.overrides:c['model_overrides']=read(args.overrides)
    assert c['rank']==4 and c['batch_size']==4
    ROOT.mkdir(parents=True,exist_ok=True);freeze(ROOT/'config.json',c)
    prompts=read(c['prompt_source'])
    needed={'train':c['train_images'],'test':c['test_images'],'calibration':c['calibration_images']}
    selected={k:prompts[k][:n] for k,n in needed.items()}
    assert all(len(selected[k])==n for k,n in needed.items())
    ids=[set(r['image_id'] for r in selected[k]) for k in needed]
    assert all(not a&b for i,a in enumerate(ids) for b in ids[i+1:]),'Prompt image IDs overlap between splits'
    freeze(ROOT/'prompts.json',selected)
    from saturday_channels import SETTINGS
    freeze(ROOT/'protocol.json',{'task':'5 native watermark teachers -> seven fixed manipulated datasets ->3 independently initialized rank4 students each',
        'train_count':105,'teacher_test_count':35,'attacks':SETTINGS,'attack_seed_rule':'20261029+i rotation/blur;20261030+i otherwise',
        'teacher_test':'manipulate existing teacher output before native decoding',
        'student_test':'fresh unmanipulated student outputs; student trained only on matching manipulated teacher images',
        'student_initialization':'clean independently downloaded model; clean VAE/text; no marked VAE inherited',
        'checkpoint_selection':'fixed final step; no detector selection; no custom quality/watermark losses',
        'prompt_sha256':sha(ROOT/'prompts.json'),'source_pins':read(REPO/'config/upstream.json'),'code':code_identity()})
    db=database();db.execute('create table if not exists tasks(id text primary key,kind text,method text,arch text,attack text,state text,worker text,updated real,error text)')
    for method in c['methods']:
        db.execute('insert or ignore into tasks values(?,?,?,?,?,?,?,?,?)',(method,'teacher',method,'','', 'pending','',0,''))
        for arch in c['students']:
            for attack in c['attacks']:
                db.execute('insert or ignore into tasks values(?,?,?,?,?,?,?,?,?)',(f'{method}/{arch}/{attack}','student',method,arch,attack,'pending','',0,''))
    db.commit();db.close()
def database():
    db=sqlite3.connect(ROOT/'queue.sqlite',timeout=60);db.execute('pragma journal_mode=WAL');return db

def decode_folder(backend,folder,attack=None,payload_output=None):
    import numpy as np,torch
    from saturday_channels import transform
    c=config();files=sorted(Path(folder).glob('sample_*.png'));coords=[];payload=[]
    if len(files)<c['test_images']:raise ValueError('Image set incomplete')
    for start in range(0,len(files),c['score_batch']):
        ids=list(range(start,min(start+c['score_batch'],len(files))));x=pixels(files[start:start+c['score_batch']])
        if attack:x=torch.stack([transform(t,attack,i) for t,i in zip(x,ids)])
        value=backend.coordinates(x,ids)
        if not np.isfinite(value).all():raise RuntimeError('Nonfinite native detector output')
        coords.extend(value.tolist())
        if hasattr(backend,'last_bits'):payload.extend(backend.last_bits.tolist())
    if payload_output and payload:
        from metrics import summarize
        freeze(payload_output,{'coordinates':payload,'metrics':summarize(np.asarray(payload)*2-1,backend.payload_key,c['aggregate_sizes'],c['partitions'],c['nominal_bit_fpr']),
            'role':'restored payload only; main table uses native fused detector'})
    return coords
def calibration(backend,method):
    from metrics import scalar_calibration
    p=ROOT/'teachers'/method
    if backend.key is not None:return None
    f=p/'scalar_calibration.json'
    if not f.exists():
        coords=decode_folder(backend,ROOT/'clean'/method/'calibration')
        c=config();thresholds=scalar_calibration(coords,c['aggregate_sizes'],c['partitions'],c['scalar_clean_fpr'])
        freeze(f,{'thresholds':thresholds,'fpr':c['scalar_clean_fpr'],'source':'disjoint unwatermarked teacher calibration; no test images',
            'calibration_images':len(coords),'decoder_identity':sha(p/'native_identity.json'),'grouping':'same per-size random partition procedure; not independent trials'})
        freeze(p/'clean_calibration_coordinates.json',coords)
    return read(f)['thresholds']

def teacher(method):
    import torch
    from backends import init,base_pipe
    from download import teacher_assets
    from metrics import summarize
    c=config();p=ROOT/'teachers'/method;p.mkdir(parents=True,exist_ok=True);status(p,'download_native_artifacts');teacher_assets(method)
    status(p,'load_native_teacher');backend=init(method);prompts=read(ROOT/'prompts.json')
    for split in ['train','test']:
        folder=p/'images'/split;folder.mkdir(parents=True,exist_ok=True);rows=prompts[split]
        # GROW author API is intentionally serial; other native methods batch.
        for start in range(0,len(rows),c['teacher_batch']):
            ids=list(range(start,min(start+c['teacher_batch'],len(rows))))
            if all((folder/image_name(i)).exists() for i in ids):continue
            seeds=[c['seed']*1000000+rows[i]['image_id'] for i in ids]
            images=backend.generate([rows[i]['text'] for i in ids],seeds);save_images(images,folder,ids);status(p,'generate',split=split,images=start+len(ids))
        manifest(folder,rows)
    if backend.key is None:
        # Create unwatermarked references with clean native teacher weights.
        clean=base_pipe(method);folder=ROOT/'clean'/method/'calibration';rows=prompts['calibration'];folder.mkdir(parents=True,exist_ok=True)
        for start in range(0,len(rows),c['teacher_batch']):
            ids=list(range(start,min(start+c['teacher_batch'],len(rows))))
            if all((folder/image_name(i)).exists() for i in ids):continue
            with torch.inference_mode():
                images=clean([rows[i]['text'] for i in ids],num_inference_steps=c['inference_steps'],guidance_scale=c['cfg'],height=512,width=512,generator=[torch.Generator('cuda').manual_seed(c['seed']*1000000+rows[i]['image_id']) for i in ids]).images
            save_images(images,folder,ids)
        manifest(folder,rows);del clean;gc.collect();torch.cuda.empty_cache()
    thresholds=calibration(backend,method)
    for attack in c['attacks']:
        f=p/'results'/f'{attack}.json'
        if f.exists():continue
        status(p,'evaluate_teacher',attack=attack)
        coordinates=decode_folder(backend,p/'images/test',attack,p/'payload'/f'{attack}.json')
        freeze(p/'coordinates'/f'{attack}.json',coordinates)
        result=summarize(coordinates,backend.key,c['aggregate_sizes'],c['partitions'],c['nominal_bit_fpr'],thresholds)
        result.update(method=method,model='teacher',attack=attack,decoder_identity=sha(p/'native_identity.json'))
        freeze(f,result)
    freeze(p/'complete.json',{'teacher':method,'identity':sha(p/'native_identity.json'),'training_images':sha(p/'images/train/image_hashes.json'),'prompts':sha(ROOT/'prompts.json')})
    status(p,'completed')
def student_job(method,arch,attack):
    from student import train,generate
    from backends import init
    from metrics import summarize
    p=ROOT/'students'/method/arch/attack;p.mkdir(parents=True,exist_ok=True)
    override=config().get('trained_student_overrides',{}).get(f'{method}/{arch}/{attack}')
    if override:
        from huggingface_hub import snapshot_download
        model=Path(snapshot_download(override['repo'],revision=override['revision'],cache_dir=str(CACHE/'hub'),allow_patterns=['adapter_model.safetensors','adapter_config.json']))
        if not (p/'adapter').exists():(p/'adapter').symlink_to(model.resolve(),target_is_directory=True)
        freeze(p/'external_student.json',{'source':override,'adapter_sha256':sha(model/'adapter_model.safetensors'),'training_provenance_supplied_by_user':True})
    else:train(method,arch,attack)
    if not (p/'images/image_hashes.json').exists():generate(method,arch,attack)
    # Release the student weights before loading the independent owner decoder.
    import gc,torch;gc.collect();torch.cuda.empty_cache()
    status(p,'evaluate');backend=init(method);coordinates=decode_folder(backend,p/'images',payload_output=p/'payload.json');freeze(p/'coordinates.json',coordinates)
    c=config();thresholds=calibration(backend,method)
    result=summarize(coordinates,backend.key,c['aggregate_sizes'],c['partitions'],c['nominal_bit_fpr'],thresholds)
    result.update(method=method,model=arch,trained_on_manipulation=attack,post_generation_manipulation=False,decoder_identity=sha(ROOT/'teachers'/method/'native_identity.json'))
    freeze(p/'results.json',result);status(p,'completed')
def claim(worker):
    db=database();db.execute('begin immediate')
    rows=db.execute("select id,kind,method,arch,attack from tasks where state='pending' order by kind desc,id").fetchall()
    chosen=None
    for row in rows:
        id,kind,method,arch,attack=row
        if kind=='student' and not (ROOT/'teachers'/method/'complete.json').exists():continue
        chosen=row;break
    if chosen:db.execute("update tasks set state='running',worker=?,updated=? where id=?",(worker,time.time(),chosen[0]))
    db.commit();db.close();return chosen
def worker():
    identity=f'{os.uname().nodename}:{os.getpid()}';status(ROOT/'workers'/identity,'idle')
    while True:
        row=claim(identity)
        if row is None:
            db=database();remaining=db.execute("select count(*) from tasks where state in ('pending','running')").fetchone()[0];db.close()
            if not remaining:return
            time.sleep(20);continue
        id,kind,method,arch,attack=row;log=ROOT/'logs'/(id.replace('/','__')+'.log');log.parent.mkdir(exist_ok=True)
        cmd=[sys.executable,str(REPO/'campaign.py'),'run-task','--method',method,'--kind',kind]
        if kind=='student':cmd+=['--student',arch,'--attack',attack]
        with log.open('a') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        db=database();db.execute('update tasks set state=?,updated=?,error=? where id=?',('completed' if result.returncode==0 else 'failed',time.time(),'' if result.returncode==0 else str(log),id));db.commit();db.close()
        from report import main
        main()
        if result.returncode==0:print('DONE',METHODS[method], 'Teacher' if kind=='teacher' else f'{STUDENTS[arch]} rank4 [{attack}]',flush=True)
        else:print('FAILED',METHODS[method],kind,arch,attack,'details:',log,flush=True)
def run_task(args):
    from diffusers.utils import logging as dl
    from transformers.utils import logging as tl
    dl.set_verbosity_error();tl.set_verbosity_error()
    if args.kind=='teacher':teacher(args.method)
    else:student_job(args.method,args.student,args.attack)
def launch(args):
    assert (ROOT/'protocol.json').exists(),'initialize first'
    for previous in read(ROOT/'launch.json') if (ROOT/'launch.json').exists() else []:
        try:os.kill(previous['pid'],0)
        except ProcessLookupError:continue
        raise RuntimeError('Workers already running; use status, do not launch duplicates')
    processes=[]
    for gpu in args.gpus.split(','):
        for slot in range(args.workers_per_gpu):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1',TQDM_DISABLE='1')
            path=ROOT/'logs'/f'worker_gpu{gpu}_{slot}.log';path.parent.mkdir(exist_ok=True)
            with path.open('a') as log:p=subprocess.Popen([sys.executable,str(REPO/'campaign.py'),'worker'],env=env,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append({'gpu':gpu,'slot':slot,'pid':p.pid,'log':str(path)})
    write(ROOT/'launch.json',processes);print('QUEUED',len(config()['methods'])*len(config()['students'])*len(config()['attacks']),'students;',len(processes),'workers')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['initialize','launch','worker','run-task','status','retry'])
    p.add_argument('--config',type=Path,default=REPO/'config/default.json');p.add_argument('--prompts',type=Path);p.add_argument('--overrides',type=Path)
    p.add_argument('--method');p.add_argument('--kind',choices=['teacher','student']);p.add_argument('--student');p.add_argument('--attack');p.add_argument('--gpus',default='0,1');p.add_argument('--workers-per-gpu',type=int,default=2)
    args=p.parse_args()
    if args.operation=='initialize':initialize(args);print('INITIALIZED')
    elif args.operation=='launch':launch(args)
    elif args.operation=='worker':worker()
    elif args.operation=='run-task':run_task(args)
    elif args.operation=='retry':
        db=database()
        if args.method:db.execute("update tasks set state='pending',error='' where state='failed' and method=?",(args.method,))
        else:db.execute("update tasks set state='pending',error='' where state='failed'")
        db.commit();db.close();print('Failed tasks reset; existing workers will retry')
    else:
        db=database();print(db.execute('select kind,state,count(*) from tasks group by kind,state').fetchall())
