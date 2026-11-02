import unittest,tempfile,json,ast
from pathlib import Path
import numpy as np
from benchmark.shared.evaluation.native_metrics import calibrate,evaluate
from benchmark.shared.data.protocol import validate
from benchmark.shared.student_training.integration import dit_source,sd14,REF,function

class GuideTests(unittest.TestCase):
    def test_native_metric_separation(self):
        key=np.ones(16);x=np.ones((2,16));x[0,0]=-1
        self.assertEqual(evaluate('grow',x,key)['value'],.5)
        self.assertEqual(evaluate('trustmark',x,key)['value'],31/32)
        with self.assertRaises(ValueError):evaluate('stable_signature',x,key)
    def test_ss_tail(self):
        r=evaluate('stable_signature',np.ones((2,48)),np.ones(48))
        self.assertEqual(r['value'],1);self.assertEqual(r['alpha'],1e-6)
        self.assertNotIn('aggregate',r)
    def test_clean_threshold_frozen_and_shared(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'threshold.json';r=calibrate(np.arange(100),method='serum',clean_manifest_sha256='abc',output=p)
            self.assertLessEqual(r['observed_calibration_fpr'],.01)
            self.assertEqual(evaluate('serum',[99],threshold=r)['value'],1)
            with self.assertRaises(ValueError):calibrate(np.arange(100)+1,method='serum',clean_manifest_sha256='abc',output=p)
            with self.assertRaises(ValueError):calibrate(np.arange(100),method='trustmark',clean_manifest_sha256='abc',output=p)
            with self.assertRaises(ValueError):evaluate('serum',[99])
    def test_gauss_needs_full_payload(self):
        t={'method':'gaussmarker','threshold':.5,'source':'independent_clean_SD2.1_teacher_only'}
        r=evaluate('gaussmarker',[.6],key=[1,0],threshold=t,payload_bits=[[1,0]])
        self.assertEqual(r['bit_accuracy'],1)
    def test_real_prompt_splits(self):
        f=Path(__file__).resolve().parents[1]/'shared/data/prompts.json';data=json.loads(f.read_text())
        r=validate(data);self.assertFalse(r['teacher_training_disjoint_verified'])
        self.assertTrue(validate(data,training_free=True)['teacher_training_disjoint_verified'])
        with self.assertRaises(ValueError):validate(data,calibration=[])
        with self.assertRaises(ValueError):validate(data,calibration=[data['test'][0]])
    def test_reused_dit_loops_compile_all_modes(self):
        for rank in [None,4,16,32]:
            s=dit_source(rank);compile(s,'integration','exec')
            self.assertIn("target=noise-z",s)
            self.assertIn("optimizer.load_state_dict",s)
            self.assertNotIn('from semantic_teacher',s)
            if rank is None:self.assertIn('model.save_pretrained',s)
            else:self.assertIn(f'LoraConfig(r={rank},lora_alpha={rank}',s)
    def test_sd_reuses_original_full_and_lora_commands(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'data').mkdir();(p/'data/image_hashes.json').write_text('[]')
            for rank in [None,4,16,32]:
                v=sd14(p/str(rank),p/'model',p/'data',rank)
                cmd=v['builder'](v['config'],v['assets'],p,v['branch'])
                self.assertEqual(cmd[cmd.index('--max_train_steps')+1],'20000')
                if rank is not None:self.assertEqual(cmd[cmd.index('--rank')+1],str(rank))
    def test_pinned_epoch_loop_patch_compatible(self):
        marker='    for epoch in range(first_epoch, args.num_train_epochs):\n'
        for name in ['train_text_to_image.py','train_text_to_image_lora.py']:
            self.assertEqual((REF/'harness/vendor/diffusers'/name).read_text().count(marker),1)

if __name__=='__main__':unittest.main()
