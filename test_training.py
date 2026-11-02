"""Real pinned Diffusers/PEFT model tests with small randomly initialized configs."""
import tempfile
import unittest
import os
from pathlib import Path
import torch
from training_checks import checked_update


def tiny_model(arch):
    from diffusers import UNet2DConditionModel,SanaTransformer2DModel,PixArtTransformer2DModel
    if arch=='sd14':
        return UNet2DConditionModel(sample_size=8,in_channels=4,out_channels=4,
            down_block_types=('CrossAttnDownBlock2D',),up_block_types=('CrossAttnUpBlock2D',),
            block_out_channels=(16,),layers_per_block=1,cross_attention_dim=16,
            attention_head_dim=4,norm_num_groups=4)
    if arch=='sana600m':
        return SanaTransformer2DModel(in_channels=4,out_channels=4,num_attention_heads=2,
            attention_head_dim=8,num_layers=1,num_cross_attention_heads=2,cross_attention_head_dim=8,
            cross_attention_dim=16,caption_channels=16,sample_size=8)
    return PixArtTransformer2DModel(in_channels=4,out_channels=8,num_attention_heads=2,
        attention_head_dim=8,num_layers=1,cross_attention_dim=16,caption_channels=16,
        sample_size=8,patch_size=2,norm_num_groups=4,use_additional_conditions=False)


def forward(model,arch,x,text,t):
    if arch=='sd14':return model(x,t,encoder_hidden_states=text).sample
    kw=dict(hidden_states=x,encoder_hidden_states=text,timestep=t,
            encoder_attention_mask=torch.ones(text.shape[:2],device=x.device),return_dict=False)
    if arch=='pixart512':kw['added_cond_kwargs']={'resolution':None,'aspect_ratio':None}
    value=model(**kw)[0]
    return value.chunk(2,dim=1)[0] if arch=='pixart512' else value


class OptimizerChecks(unittest.TestCase):
    def test_nonfinite_gradients_never_reach_optimizer(self):
        p=torch.nn.Parameter(torch.tensor([0.]))
        opt=torch.optim.AdamW([p],lr=.1)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        # Finite loss, infinite derivative. This previously escaped loss-only checks.
        with self.assertRaisesRegex(RuntimeError,'Nonfinite unscaled'):
            checked_update(p.sqrt().sum(),opt,scaler,[p])
        self.assertEqual(p.item(),0.)
        self.assertFalse(opt.state)
    def test_large_finite_gradients_do_not_overflow_the_clipping_norm(self):
        p=torch.nn.Parameter(torch.tensor([1.,1.]))
        opt=torch.optim.AdamW([p],lr=.1)
        result=checked_update((p*1e30).sum(),opt,torch.amp.GradScaler('cuda',enabled=False),[p])
        self.assertEqual(result['skipped'],0)
        self.assertTrue(torch.isfinite(p).all())
        self.assertGreater(result['grad_norm'],1e30)
    def test_nonfinite_loss_is_rejected_before_backward(self):
        p=torch.nn.Parameter(torch.tensor([1.]))
        opt=torch.optim.AdamW([p],lr=.1)
        with self.assertRaisesRegex(RuntimeError,'Nonfinite loss'):
            checked_update((p*float('nan')).sum(),opt,torch.amp.GradScaler('cuda',enabled=False),[p])
        self.assertIsNone(p.grad)
    def test_amp_overflow_is_skipped_then_a_finite_step_can_recover(self):
        device=os.environ.get('BASELINE_TEST_DEVICE','cpu')
        p=torch.nn.Parameter(torch.tensor([0.],device=device))
        opt=torch.optim.AdamW([p],lr=.1)
        scaler=torch.amp.GradScaler(device,init_scale=1024.)
        update=checked_update(p.sqrt().sum(),opt,scaler,[p])
        self.assertEqual(update['skipped'],1)
        self.assertEqual(p.item(),0.)
        self.assertFalse(opt.state)
        self.assertEqual(scaler.get_scale(),512.)
        update=checked_update((p-1).square().sum(),opt,scaler,[p])
        self.assertEqual(update['skipped'],0)
        self.assertGreater(p.item(),0.)


class NativeAdapterTest(unittest.TestCase):
    def test_forward_backward_save_reload_for_all_architectures(self):
        from student import add_lora,load_adapter
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file
        torch.set_num_threads(2)
        for arch in ['sd14','sana600m','pixart512']:
            with self.subTest(arch=arch):
                torch.manual_seed(123)
                device=os.environ.get('BASELINE_TEST_DEVICE','cpu')
                dtype=(torch.float16 if arch=='sd14' else torch.bfloat16) if device=='cuda' else torch.float32
                model=tiny_model(arch).to(device=device,dtype=dtype)
                base={n:p.clone() for n,p in model.state_dict().items()}
                cfg,_=add_lora(model,arch)
                frozen=[(p,p.detach().clone()) for p in model.parameters() if not p.requires_grad]
                params=[p for p in model.parameters() if p.requires_grad]
                self.assertTrue(params and all(p.dtype==torch.float32 for p in params))
                model.train();model.enable_gradient_checkpointing()
                x=torch.randn(2,4,8,8,device=device,dtype=dtype);text=torch.randn(2,4,16,device=device,dtype=dtype);t=torch.tensor([100,200],device=device)
                opt=torch.optim.AdamW(params,lr=.001)
                scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda' and arch=='sd14',init_scale=1024.)
                for _ in range(3):
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device,dtype=dtype,enabled=device=='cuda'):
                        pred=forward(model,arch,x,text,t)
                        loss=pred.float().square().mean()
                    result=checked_update(loss,opt,scaler,params)
                    self.assertEqual(result['skipped'],0)
                for p,expected in frozen:torch.testing.assert_close(p,expected,rtol=0,atol=0)
                self.assertTrue(any(p.abs().max()>0 for n,p in model.named_parameters() if 'lora_B' in n))
                model.eval()
                with torch.no_grad():expected=forward(model,arch,x,text,t)
                with tempfile.TemporaryDirectory() as tmp:
                    cfg.save_pretrained(tmp)
                    save_file({n:p.detach().cpu().contiguous() for n,p in get_peft_model_state_dict(model).items()},str(Path(tmp)/'adapter_model.safetensors'))
                    restored=tiny_model(arch).to(device=device,dtype=dtype);restored.load_state_dict(base)
                    load_adapter(restored,tmp)
                    with torch.no_grad():actual=forward(restored,arch,x,text,t)
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    with self.assertRaisesRegex(RuntimeError,'already has an adapter'):load_adapter(restored,tmp)


if __name__=='__main__':unittest.main()
