"""CPU preparation/reporting by default. GPU stages require --execute explicitly."""
import argparse,json,sys,os,subprocess,hashlib
from pathlib import Path
from benchmark.shared.data.protocol import validate,digest
BASE=Path(__file__).resolve().parent

def dump(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2))
def read(p):return json.loads(Path(p).read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['plan','verify-sources','generate','score','calibrate','evaluate','student-train','student-generate'])
    p.add_argument('--method',choices=['stable_signature','gaussmarker','trustmark','grow','serum'])
    p.add_argument('--work',type=Path,default=Path('guide_work'));p.add_argument('--artifacts',type=Path)
    p.add_argument('--prompts',type=Path,default=BASE/'shared/data/prompts.json');p.add_argument('--calibration',type=Path)
    p.add_argument('--teacher-training',type=Path);p.add_argument('--assets',type=Path);p.add_argument('--input',type=Path)
    p.add_argument('--threshold',type=Path);p.add_argument('--output',type=Path);p.add_argument('--model',type=Path)
    p.add_argument('--arch',choices=['sd14','sana600m','pixart512']);p.add_argument('--mode',choices=['full','r4','r16','r32'])
    p.add_argument('--split',choices=['train','validation','test','calibration'],default='test')
    p.add_argument('--attack',default='clean');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.stage=='verify-sources':
        manifest=read(BASE/'configs/official_sources.json')
        for name,m in manifest.items():
            for rel,h in m['files'].items():
                if digest(BASE/'methods'/name/'official'/rel)!=h:raise ValueError('Author source changed: '+name+'/'+rel)
        for rel,h in read(BASE/'configs/student_sources.json').items():
            if digest(BASE/'shared/student_training/reference'/rel)!=h:raise ValueError('Student reference changed: '+rel)
        print('PASS official/student source hashes');return
    if a.stage=='plan':
        prompts=read(a.prompts);cal=read(a.calibration) if a.calibration else None
        report=validate(prompts,cal,read(a.teacher_training) if a.teacher_training else None,a.method=='grow')
        report['prompt_sha256']=digest(a.prompts)
        report['jobs']=[{'method':method,'arch':arch,'mode':mode} for method in read(BASE/'configs/benchmark.json')['methods'] for arch in ['sd14','sana600m','pixart512'] for mode in ['full','r32','r16','r4']]
        report['empirical_methods_ready']=cal is not None
        dump(a.work/'plan.json',report);print('PREPARED 60 student configurations; no GPU/network execution');return
    if a.stage=='calibrate':
        from benchmark.shared.evaluation.native_metrics import calibrate
        data=read(a.input)
        if data.get('model')!='clean_teacher' or data.get('method')!=a.method:raise ValueError('Clean teacher score provenance required')
        calibrate(data['scores'],method=a.method,clean_manifest_sha256=data['image_manifest_sha256'],output=a.output,attack=data['attack']);print('FROZEN threshold');return
    if a.stage=='evaluate':
        from benchmark.shared.evaluation.native_metrics import evaluate
        data=read(a.input);threshold=read(a.threshold) if a.threshold else None
        if data['method']!=a.method:raise ValueError('Method mismatch')
        if threshold and threshold['attack']!=data['attack']:raise ValueError('Manipulation/calibration mismatch')
        result=evaluate(a.method,data['coordinates'],data.get('key'),threshold,data.get('payload_bits'))
        dump(a.output,{'method':a.method,'model':data['model'],'attack':data['attack'],**result});print('DONE evaluation',a.method,data['model']);return
    if not a.execute:raise ValueError('GPU stage is preparation-only unless --execute is explicitly supplied')
    if a.stage=='student-train':
        if not all([a.arch,a.mode,a.model,a.input]):raise ValueError('Need --arch --mode --model --input baseline data source')
        if a.arch!='sd14':
            subprocess.run([sys.executable,'-m','benchmark.shared.student_training.worker','--job',str(a.work),'--source',str(a.input),'--model',str(a.model),'--arch',a.arch,'--mode',a.mode],check=True);return
        from benchmark.shared.student_training.integration import sd14
        job=a.work.resolve();data=a.input.resolve()/'images/teacher_wm_train';rank=None if a.mode=='full' else int(a.mode[1:])
        v=sd14(job,a.model,data,rank);target=job/'images/teacher_wm_train';target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.symlink_to(data,target_is_directory=True)
        cmd=v['builder'](v['config'],v['assets'],job,v['branch']);cmd[cmd.index('--checkpointing_steps')+1]='1250'
        subprocess.run([sys.executable,str(v['helper']/'driver.py'),'--trainer',cmd[1],'--identity-file',str(v['identity']),*cmd[2:]],check=True);return
    if a.stage=='student-generate':
        from benchmark.shared.evaluation.student_images import generate
        generate(a.arch,a.mode,a.model,a.work,read(a.prompts)[a.split],a.output);return
    if not all([a.method,a.artifacts,a.assets]):raise ValueError('Need --method --artifacts --assets for native GPU stages')
    # Use prepared local artifacts only. Author code itself remains unchanged.
    os.environ['BASELINE_WORK']=str(a.work.resolve());os.environ['BASELINE_CACHE']=str(a.artifacts.resolve())
    assets=read(a.assets);cfg={**read(Path(__file__).resolve().parents[1]/'config/default.json'),**assets}
    cfg.update(inference_steps=50,cfg=7.5,teacher_path=assets['teacher_path']);dump(a.work/'config.json',cfg)
    dump(a.artifacts/'models/teacher.json',{'path':assets['teacher_path']})
    from benchmark.methods.native import create
    from common import pixels,save_images,manifest
    if a.stage=='generate':
        if a.split=='calibration':
            if not a.calibration:raise ValueError('Independent calibration prompts required')
            rows=read(a.calibration);validate(read(a.prompts),rows)
        else:rows=read(a.prompts)[a.split]
        model=create(a.method,a.work)
        folder=a.output;folder.mkdir(parents=True,exist_ok=True)
        for i,row in enumerate(rows):
            if a.split=='calibration':
                import torch
                with torch.inference_mode():images=model.pipe(row['text'],height=512,width=512,num_inference_steps=50,guidance_scale=7.5,generator=torch.Generator('cuda').manual_seed(int(row['image_id']))).images
            else:images=model.generate([row['text']],[int(row['image_id'])])
            save_images(images,folder,[i])
        manifest(folder,rows);dump(folder/'generation_provenance.json',{'role':'clean_teacher' if a.split=='calibration' else 'watermarked_teacher','method':a.method,'model':str(getattr(model.pipe.config,'_name_or_path',cfg['teacher_path'])),'prompt_sha256':digest(a.calibration if a.split=='calibration' else a.prompts)});print('DONE generation',a.method,a.split);return
    model=create(a.method,a.work);files=sorted(a.input.glob('sample_*.png'));coordinates=[];payload=[]
    if a.split=='calibration':
        provenance=read(a.input/'generation_provenance.json')
        if provenance.get('role')!='clean_teacher' or provenance['method']!=a.method:raise ValueError('Only clean teacher generation may be calibrated')
    if len(files)!=1000 and a.split!='calibration':raise ValueError('Need exactly1000 test images')
    from benchmark.shared.manipulations.saturday import transform
    for i,f in enumerate(files):
        x=pixels([f])
        if a.attack!='clean':x=transform(x[0],a.attack,i)[None]
        coordinates.extend(model.coordinates(x,[i]).tolist())
        if hasattr(model,'last_bits'):payload.extend(model.last_bits.tolist())
    dump(a.output,{'method':a.method,'model':'clean_teacher' if a.split=='calibration' else str(a.input.resolve()),
        'attack':a.attack,'image_manifest_sha256':digest(a.input/'image_hashes.json'),'coordinates':coordinates,
        'scores':[v[0] for v in coordinates] if a.method in ['serum','gaussmarker'] else None,
        'key':getattr(model,'key',None),'payload_bits':payload or None})
    print('DONE score',a.method,a.input)

if __name__=='__main__':main()
