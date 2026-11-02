"""Thin native-method adapters. Teacher watermark algorithms are not optimized here."""
from pathlib import Path
import sys,json,types,math
import numpy as np
import torch
from PIL import Image
from common import ROOT,CACHE,config,read,freeze,upstream,policy,sha

def backbone(method):
    return read(CACHE/'teachers/serum/backbone.json')['path'] if method=='serum' else read(CACHE/'models/teacher.json')['path']
def base_pipe(method,dtype=torch.float16):
    from diffusers import StableDiffusionPipeline
    pipe=StableDiffusionPipeline.from_pretrained(backbone(method),torch_dtype=dtype,safety_checker=None,requires_safety_checker=False,local_files_only=True).to('cuda')
    if method=='gaussmarker':
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True);pipe.unet.eval().requires_grad_(False);pipe.vae.eval().requires_grad_(False);pipe.text_encoder.eval().requires_grad_(False)
    return pipe
def pil(x):return Image.fromarray((x.detach().cpu().permute(1,2,0).clamp(0,1).numpy()*255).astype('uint8'))
def init(method):
    policy();c=config();p=ROOT/'teachers'/method;p.mkdir(parents=True,exist_ok=True)
    if method=='stable_signature':return StableSignature(p,c)
    if method=='trustmark':return TrustMark(p,c)
    if method=='grow':return Grow(p,c)
    if method=='serum':return Serum(p,c)
    if method=='gaussmarker':return GaussMarker(p,c)
    raise ValueError(method)
class Native:
    key=None
    def generate(self,prompts,seeds):
        with torch.inference_mode():return self.pipe(prompts,height=512,width=512,num_inference_steps=self.c['inference_steps'],guidance_scale=self.c['cfg'],generator=[torch.Generator('cuda').manual_seed(s) for s in seeds]).images

class StableSignature(Native):
    def __init__(self,p,c):
        from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
        self.p=p;self.c=c;self.pipe=base_pipe('stable_signature',torch.float32)
        path=CACHE/'teachers/stable_signature/sd2_decoder.pth'
        state=torch.load(path,map_location='cpu',weights_only=True)
        if 'ldm_decoder' in state:state=state['ldm_decoder']
        converted=convert_ldm_vae_checkpoint(state,dict(self.pipe.vae.config))
        marked={k:v for k,v in converted.items() if k.startswith(('decoder.','post_quant_conv.'))}
        result=self.pipe.vae.load_state_dict(marked,strict=False)
        assert len(marked)>100 and not result.unexpected_keys
        assert not any(k.startswith(('decoder.','post_quant_conv.')) for k in result.missing_keys)
        self.pipe.to(torch.float16)
        self.decoder=torch.jit.load(str(CACHE/'teachers/stable_signature/extractor.pt'),map_location='cuda').eval()
        self.key=[int(x) for x in '111010110101000001010111010011010100010000100111']
        freeze(p/'native_identity.json',{'method':'Stable Signature published SD2 decoder + standard whitened 48bit extractor','decoder_sha256':sha(path),'key':self.key,'preprocessing':'ImageNet mean/std; raw512 RGB [0,1]; logits>0','base':backbone('stable_signature')})
    def coordinates(self,x,indices):
        mean=x.new_tensor([.485,.456,.406]).view(1,3,1,1);std=x.new_tensor([.229,.224,.225]).view(1,3,1,1)
        with torch.inference_mode():return self.decoder(((x-mean)/std).cuda()).float().cpu().numpy()

class TrustMark(Native):
    def __init__(self,p,c):
        sys.path.insert(0,str(upstream('trustmark')/'python'))
        from trustmark import TrustMark as TM
        self.p=p;self.c=c;self.pipe=base_pipe('trustmark');self.tm=TM(use_ECC=True,verbose=False,model_type=c['trustmark_model_type'],loadRemover=False,device='cuda')
        self.message=c['trustmark_message'];self.key=self.tm.ecc.encode_text([self.message])[0].astype(int).tolist()
        freeze(p/'native_identity.json',{'method':'TrustMark-Q official encoder/decoder, default BCH5','payload_text':self.message,'coded_key':self.key,'metric_bits':'100 coded bits, not independent payload bits','preprocessing':'native RGB8 PIL; native square processing;245px bilinear;[-1,1]','base':backbone('trustmark')})
    def generate(self,prompts,seeds):return [self.tm.encode(im,self.message,MODE='text') for im in super().generate(prompts,seeds)]
    def coordinates(self,x,indices):
        from torchvision.transforms.functional import to_tensor
        ims=[to_tensor(self.tm.get_the_image_for_processing(pil(t)).resize((self.tm.model_resolution_dec,)*2,Image.Resampling.BILINEAR))*2-1 for t in x]
        with torch.inference_mode():return self.tm.decoder.decoder(torch.stack(ims).cuda()).float().cpu().numpy()

class Grow(Native):
    def __init__(self,p,c):
        sys.path.insert(0,str(upstream('GROW')/'GROW'))
        from grow import GrowConfig,GrowWatermarker
        from diffusers import DDIMScheduler
        self.p=p;self.c=c;self.pipe=base_pipe('grow',torch.float32);self.pipe.scheduler=DDIMScheduler.from_config(self.pipe.scheduler.config)
        cfg=GrowConfig(model_id=backbone('grow'),message=c['grow_message'],secret_key=c['grow_secret_key'],dtype='float32',num_inference_steps=c['inference_steps'],cfg_scale=c['cfg'])
        self.wm=GrowWatermarker(cfg,pipe=self.pipe)
        from grow.codec import message_to_bits
        self.key=message_to_bits(cfg.message)
        freeze(p/'native_identity.json',{'method':'GROW upstream default progressive guidance; no robustness-search decoding','config':vars(cfg),'key':self.key,'coordinate':'native within-image majority vote, first-seen tie break; hard-bit votes across images; no oracle search'})
    def generate(self,prompts,seeds):return [self.wm.generate_with_watermark(prompt,seed) for prompt,seed in zip(prompts,seeds)]
    def coordinates(self,x,indices):
        from grow.utils import extract_center_patch,dct2d_torch
        cfg=self.wm.cfg
        with torch.no_grad():
            z=self.pipe.vae.encode((x.cuda()*2-1).to(self.wm.dtype)).latent_dist.mean*self.pipe.vae.config.scaling_factor
            z=dct2d_torch(extract_center_patch(z,cfg.center_ratio))
        h,w=z.shape[-2:];coords=[(i,j) for i in range(int(h*cfg.dct_min),int(h*cfg.dct_max)) for j in range(int(w*cfg.dct_min),int(w*cfg.dct_max))]
        np.random.default_rng(sum(map(ord,cfg.secret_key))).shuffle(coords)
        columns=[];n=len(self.key);channels=len(cfg.channels)
        for i,ch in enumerate(cfg.channels):
            k=n//channels+(i<n%channels)
            if not k:continue
            reps=len(coords)//k
            # Native _read_bits uses majority voting, with first-seen value on
            # ties (Counter.most_common). Preserve that per-image decoder.
            raw=torch.stack([z[:,ch,a,b] for a,b in coords[:reps*k]],dim=1).reshape(len(x),reps,k)>0
            counts=raw.sum(1);decoded=torch.where(counts*2==reps,raw[:,0,:],counts*2>reps)
            columns.append(decoded.float()*2-1)
        return torch.cat(columns,dim=1).float().cpu().numpy()

class Serum(Native):
    def __init__(self,p,c):
        sys.path.insert(0,str(upstream('SERUM')))
        from src.utils.config import Config
        from src.models.watermark import Watermark
        from src.models.score_model import WatermarkScoreModel
        self.p=p;self.c=c;self.pipe=base_pipe('serum')
        cfg=Config(str(upstream('SERUM')/'configs/config_sd21.yaml'))
        state=torch.load(CACHE/'teachers/serum/checkpoint.pth',map_location='cpu',weights_only=True)
        assert state['epoch']==50
        self.wm=Watermark(cfg).eval();self.detector=WatermarkScoreModel(cfg).eval()
        self.wm.load_state_dict(state['watermark_state_dict'],strict=True);self.detector.load_state_dict(state['score_model_state_dict'],strict=True)
        self.wm.cuda().requires_grad_(False);self.detector.cuda().requires_grad_(False)
        freeze(p/'native_identity.json',{'method':'SERUM author epoch50 checkpoint, fixed author SD2.1 backbone','checkpoint_sha256':sha(CACHE/'teachers/serum/checkpoint.pth'),'alpha':c['serum_alpha'],'decoder':'binary detector; posterior.sample seed42+index; no bit/key accuracy','base':backbone('serum')})
    def generate(self,prompts,seeds):
        zs=[]
        with torch.inference_mode():
            for s in seeds:
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(s);zs.append(self.wm(1,alpha=self.c['serum_alpha']))
            return self.pipe(prompts,latents=torch.cat(zs).half(),num_inference_steps=self.c['inference_steps'],guidance_scale=self.c['cfg'],height=512,width=512,generator=[torch.Generator('cuda').manual_seed(s) for s in seeds]).images
    def coordinates(self,x,indices):
        with torch.inference_mode():
            dist=self.pipe.vae.encode(x.cuda().half()*2-1).latent_dist
            from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
            zs=[DiagonalGaussianDistribution(v[None]).sample(generator=torch.Generator('cuda').manual_seed(42+i)) for v,i in zip(dist.parameters,indices)]
            return self.detector(torch.cat(zs).float()*self.pipe.vae.config.scaling_factor).float().cpu().numpy().reshape(-1,1)

class GaussMarker(Native):
    """Published dual-domain/GNR/fuser path; also retain raw restored payload bits."""
    def __init__(self,p,c):
        sys.path.insert(0,str(upstream('GaussMarker')))
        from watermark import Gaussian_Shading_chacha
        from unet.unet_model import UNet
        from diffusers import DPMSolverMultistepScheduler
        from tr_utils import get_watermarking_mask
        import joblib
        self.p=p;self.c=c;self.pipe=base_pipe('gaussmarker')
        self.pipe.scheduler=DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        files=read(CACHE/'teachers/gaussmarker/files.json');info=torch.load(files['w1_256.pth'],map_location='cpu',weights_only=False)
        self.wm=Gaussian_Shading_chacha(1,8,8,.01,1,watermark=info['w'],m=info['m'],key=info['key'],nonce=info['nonce'])
        self.m=torch.as_tensor(self.wm.m).reshape(1,4,64,64).float().cuda();self.patch=torch.load(files['w2.pth'],map_location='cpu',weights_only=False).cuda()
        self.args=types.SimpleNamespace(w_channel=3,w_mask_shape='circle',w_radius=4,w_measurement='l1_complex',w_injection='complex')
        self.mask=get_watermarking_mask(self.patch.real,self.args,'cuda')
        state=torch.load(files['model_final.pth'],map_location='cpu',weights_only=True)
        if 'state_dict' in state:state=state['state_dict']
        first=state['inc.double_conv.0.weight'];self.gnr=UNet(first.shape[1],4,nf=first.shape[0]).cuda().eval().requires_grad_(False);self.gnr.load_state_dict(state,strict=True)
        self.conditional=first.shape[1]==8;self.fuser=joblib.load(files['classifier']);self.payload_key=self.wm.watermark.flatten().int().tolist()
        # Comparable bit-level summary uses actual restored native payload bits.
        # Keep the published fuser available as an explicit optional protocol.
        self.metric=c.get('gaussmarker_metric','payload_bits')
        if self.metric not in ['payload_bits','fused_score']:raise ValueError(self.metric)
        if self.metric=='payload_bits':self.key=self.payload_key
        freeze(p/'native_identity.json',{'method':'GaussMarker 256bits, native dual-domain + GNR + published SD2.1 classifier','artifacts':{k:sha(v) for k,v in files.items()},'key':self.payload_key,'GNR_conditional':self.conditional,'metric':self.metric,'inversion':'upstream DDIM algebra on native DPM timestep grid, empty prompt, CFG1, 50steps, final alpha1'})
    def generate(self,prompts,seeds):
        from tr_utils import inject_watermark
        import random
        latents=[]
        for s in seeds:
            # Official sampling uses SciPy/NumPy truncated normals, not torch latents.
            state=np.random.get_state();np.random.seed(s)
            try:z,_=self.wm.create_watermark_and_return_w_m()
            finally:np.random.set_state(state)
            latents.append(inject_watermark(z.float().cuda(),self.mask,self.patch,self.args).half())
        with torch.inference_mode():return self.pipe(prompts,latents=torch.cat(latents),height=512,width=512,num_inference_steps=self.c['inference_steps'],guidance_scale=self.c['cfg']).images
    def coordinates(self,x,indices):
        with torch.inference_mode():
            dist=self.pipe.vae.encode(x.cuda().half()*2-1).latent_dist
            from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
            z=torch.cat([DiagonalGaussianDistribution(v[None]).sample(generator=torch.Generator('cuda').manual_seed(i)) for v,i in zip(dist.parameters,indices)])*.18215
            toks=self.pipe.tokenizer(['']*len(x),padding='max_length',max_length=77,return_tensors='pt');emb=self.pipe.text_encoder(toks.input_ids.cuda())[0]
            sch=self.pipe.scheduler;sch.set_timesteps(self.c['inference_steps']);z=z*sch.init_noise_sigma
            for t in reversed(sch.timesteps.to('cuda')):
                eps=self.pipe.unet(sch.scale_model_input(z,t),t,encoder_hidden_states=emb).sample
                prev=t-sch.config.num_train_timesteps//sch.num_inference_steps
                alpha_t=sch.alphas_cumprod[prev] if prev>=0 else sch.alphas_cumprod.new_tensor(1.)
                alpha_next=sch.alphas_cumprod[t]
                z=alpha_next.sqrt()*((alpha_t.rsqrt()-alpha_next.rsqrt())*z+((1/alpha_next-1).sqrt()-(1/alpha_t-1).sqrt())*eps)+z
            m=(z>0).float();inp=torch.cat([self.m.expand(len(x),-1,-1,-1),m],1) if self.conditional else m
            restored=(self.gnr(inp.float()).sigmoid()>.5).int().cpu()
            bits=[self.wm.pred_w_from_m(t[None]).flatten().numpy() for t in restored]
            self.last_bits=np.asarray(bits)
            if self.metric=='payload_bits':return self.last_bits.astype(np.float32)*2-1
            acc=(self.last_bits==np.asarray(self.payload_key)).mean(1)
            fft=torch.fft.fftshift(torch.fft.fft2(z),dim=(-1,-2));freq=[]
            for item in fft:freq.append(-float((item[None][self.mask]-self.patch[self.mask]).abs().mean())*.01)
            score=self.fuser.predict_proba(np.stack([acc,freq],1))[:,1]
            return score[:,None]
