"""Integration only: watermark generation/readers come from vendored author code.

Caller prepares local artifacts. This module never downloads model weights.
Each method must run in its own process/environment (author versions differ).
"""
import sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent

def official(name):
    p=BASE/name/'official'
    if not p.exists():raise FileNotFoundError(p)
    return p

def create(method,workspace):
    import common
    import backends
    backends.upstream=official
    # Legacy format conversion/loading helpers only, not legacy metrics/training.
    from backends import StableSignature,Serum,base_pipe,Native,pil
    import torch
    from common import config
    c=config();p=Path(workspace)/'native_identity';p.mkdir(parents=True,exist_ok=True)
    if method=='stable_signature':return StableSignature(p,c)
    if method=='serum':
        sys.path.insert(0,str(official('SERUM')))
        model=Serum(p,c)
        def coordinates(x,indices):
            with torch.inference_mode():
                z=model.pipe.vae.encode(x.cuda().half()*2-1).latent_dist.sample()*model.pipe.vae.config.scaling_factor
                return model.detector(z.float()).cpu().numpy().reshape(-1,1)
        model.coordinates=coordinates
        return model
    if method=='trustmark':
        sys.path.insert(0,str(official('trustmark')/'python'))
        from trustmark import TrustMark
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        model=Native();model.c=c;model.pipe=base_pipe(method)
        model.tm=TrustMark(use_ECC=False,verbose=False,model_type='Q',loadRemover=False,device='cuda')
        model.key=np.random.default_rng(0).integers(0,2,100).tolist()
        def generate(prompts,seeds):
            images=Native.generate(model,prompts,seeds)
            return [model.tm.encode(i,''.join(map(str,model.key)),MODE='binary') for i in images]
        def coordinates(x,indices):
            batch=[to_tensor(model.tm.get_the_image_for_processing(pil(t)).resize((245,245),Image.Resampling.BILINEAR))*2-1 for t in x]
            with torch.inference_mode():return model.tm.decoder.decoder(torch.stack(batch).cuda()).float().cpu().numpy()
        model.generate=generate;model.coordinates=coordinates;return model
    if method=='grow':
        sys.path.insert(0,str(official('GROW')/'GROW'))
        from grow import GrowConfig,GrowWatermarker
        from grow.codec import message_to_bits
        from diffusers import DDIMScheduler
        model=Native();model.c=c;model.pipe=base_pipe(method,torch.float32)
        model.pipe.scheduler=DDIMScheduler.from_config(model.pipe.scheduler.config)
        cfg=GrowConfig(model_id=c['teacher_path'],message='OK',dtype='float32',num_inference_steps=50,
                cfg_scale=7.5,watermark_strength=.5,guidance_scale=200.,watermark_start_ratio=.5)
        wm=GrowWatermarker(cfg,pipe=model.pipe);model.key=message_to_bits('OK')
        model.generate=lambda prompts,seeds:[wm.generate_with_watermark(t,s) for t,s in zip(prompts,seeds)]
        model.coordinates=lambda x,indices:np.asarray([wm._read_bits(pil(i)) for i in x])*2-1
        return model
    if method=='gaussmarker':return gaussmarker(c)
    raise ValueError(method)

def gaussmarker(c):
    import torch,joblib
    sys.path.insert(0,str(official('GaussMarker')))
    from inverse_stable_diffusion import InversableStableDiffusionPipeline
    from diffusers import DPMSolverMultistepScheduler
    from watermark import Gaussian_Shading_chacha
    from tr_utils import get_watermarking_mask,inject_watermark,eval_watermark
    from unet.unet_model import UNet
    from utils import set_random_seed
    from types import SimpleNamespace
    scheduler=DPMSolverMultistepScheduler.from_pretrained(c['teacher_path'],subfolder='scheduler',local_files_only=True)
    pipe=InversableStableDiffusionPipeline.from_pretrained(c['teacher_path'],scheduler=scheduler,torch_dtype=torch.float16,local_files_only=True).to('cuda')
    pipe.safety_checker=None;pipe.set_progress_bar_config(disable=True)
    info=torch.load(c['gaussmarker']['w1'],map_location='cpu',weights_only=False)
    wm=Gaussian_Shading_chacha(1,8,8,.01,1,watermark=info['w'],m=info['m'],key=info['key'],nonce=info['nonce'])
    patch=torch.load(c['gaussmarker']['w2'],map_location='cpu',weights_only=False).cuda()
    args=SimpleNamespace(w_channel=3,w_mask_shape='circle',w_radius=4,w_measurement='l1_complex',w_injection='complex')
    mask=get_watermarking_mask(patch.real,args,'cuda')
    state=torch.load(c['gaussmarker']['gnr'],map_location='cpu',weights_only=True)
    first=state['inc.double_conv.0.weight'];gnr=UNet(first.shape[1],4,nf=first.shape[0]).cuda().eval();gnr.load_state_dict(state,strict=True)
    fuser=joblib.load(c['gaussmarker']['classifier']);conditional=first.shape[1]==8
    model=SimpleNamespace(pipe=pipe,key=wm.watermark.flatten().tolist())
    def generate(prompts,seeds):
        images=[]
        for prompt,seed in zip(prompts,seeds):
            set_random_seed(seed);z,_=wm.create_watermark_and_return_w_m()
            z=inject_watermark(z.float().cuda(),mask,patch,args).half()
            images.append(pipe(prompt,latents=z,height=512,width=512,num_inference_steps=50,guidance_scale=7.5).images[0])
        return images
    def coordinates(x,indices):
        features=[];bits=[]
        with torch.no_grad():
            emb=pipe.get_text_embedding('')
            for rgb,seed in zip(x,indices):
                set_random_seed(seed)
                z=pipe.get_image_latents((rgb[None].cuda()*2-1).to(emb.dtype))
                z=pipe.forward_diffusion(latents=z,text_embeddings=emb,guidance_scale=1.,num_inference_steps=50)
                raw=(z>0).float()
                inp=torch.cat([torch.as_tensor(wm.m).reshape(1,4,64,64).float().cuda(),raw],1) if conditional else raw
                restored=(gnr(inp).sigmoid()>.5).int().cpu()
                payload=wm.pred_w_from_m(restored);bits.append(payload.flatten().numpy())
                _,freq=eval_watermark(z,z,mask,patch,args)
                features.append([float((payload.cpu()==wm.watermark.cpu()).float().mean()),-float(freq)])
        model.last_bits=np.asarray(bits)
        return fuser.predict_proba(np.asarray(features))[:,1,None]
    model.generate=generate;model.coordinates=coordinates
    return model
