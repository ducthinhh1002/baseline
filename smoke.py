"""Bounded GPU checks in an isolated work directory, before a long campaign."""
import argparse,os,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('kind',choices=['native','student']);p.add_argument('name');args=p.parse_args()
project=Path(__file__).resolve().parent
campaign_root=Path(os.environ.get('BASELINE_WORK',project/'work'))
os.environ['BASELINE_WORK']=str(campaign_root.parent/'smoke'/os.environ.get('BASELINE_PREFLIGHT_ID','manual')/args.kind/args.name)
from common import ROOT,CACHE,read,write,config,pixels,image_name,save_images,manifest
import numpy as np
import torch
if args.kind=='native':
    from download import teacher_assets
    teacher_assets(args.name)
    from backends import init
    backend=init(args.name)
    images=backend.generate(['a red bicycle beside a brick wall'],[12345]);folder=ROOT/'images';save_images(images,folder,[0])
    value=backend.coordinates(pixels([folder/image_name(0)]),[0]);assert np.isfinite(value).all()
    output={'method':args.name,'shape':list(value.shape),'score':value.tolist()}
    if backend.key is not None:
        output['clean_bit_accuracy']=float(((value>0)==np.asarray(backend.key)).mean())
    if args.name=='grow':
        native=backend.wm._read_bits(images[0]);assert np.array_equal(value[0]>0,native),'GROW native reader mismatch'
    write(ROOT/'passed.json',output);print('SMOKE PASS',output,flush=True)
else:
    from common import REPO
    c=read(campaign_root/'config.json') if (campaign_root/'config.json').exists() else config()
    c.update(train_images=4,test_images=2,calibration_images=4,steps=8,checkpoint_every=4)
    write(ROOT/'config.json',c)
    source=read(Path(os.environ.get('BASELINE_PROMPTS',c['prompt_source'])))
    rows={'train':source['train'][:4],'test':source['test'][:2],'calibration':source['calibration'][:4]};write(ROOT/'prompts.json',rows)
    folder=ROOT/'teachers/stable_signature/images/train'
    from PIL import Image
    # Real marked samples when available, otherwise a deterministic synthetic
    # fixture. This checks execution/gradients/reload, not transfer performance.
    for i in range(4):
        sample=project/'smoke/native/stable_signature/images'/image_name(0)
        image=Image.open(sample).convert('RGB') if sample.exists() else Image.fromarray(np.random.default_rng(i).integers(0,256,(512,512,3),dtype=np.uint8))
        save_images([image],folder,[i])
    manifest(folder,rows['train'])
    from student import train,generate
    train('stable_signature',args.name,'rotation15');generate('stable_signature',args.name,'rotation15')
    write(ROOT/'passed.json',{'arch':args.name,'trained':read(ROOT/'students/stable_signature'/args.name/'rotation15/trained.json'),'generated':2})
    print('SMOKE PASS student',args.name,flush=True)
