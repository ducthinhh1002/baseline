import unittest
import numpy as np
from metrics import summarize,bit_threshold,groups,scalar_calibration
class MetricsTest(unittest.TestCase):
    def test_32bit_threshold_matches_previous_experiments(self):
        self.assertEqual(bit_threshold(32,.0035001833457499743),24)
    def test_native_bits_differ_from_exact_key(self):
        x=np.ones((256,32));x[:,-8:]=-1
        r=summarize(x,[1]*32,sizes=(1,256))['rows']
        self.assertEqual(r[0]['acceptance'],1)
        self.assertEqual(r[0]['exact_full_key_rate'],0)
        self.assertEqual(r[1]['bit_accuracy'],.75)
    def test_group_counts_and_no_repeated_image_within_group(self):
        g=groups(1000,256);self.assertEqual(sum(len(x) for x in g),300)
        self.assertTrue(all(len(set(row))==256 for a in g for row in a))
    def test_binary_detector_never_reports_bit_accuracy(self):
        thresholds=scalar_calibration(np.linspace(0,1,1000),(1,32,256))
        rows=summarize(np.ones((1000,1)),sizes=(1,32,256),scalar_thresholds=thresholds)['rows']
        self.assertTrue(all(x['bit_accuracy'] is None for x in rows))
    def test_no_nan_or_missing_threshold(self):
        with self.assertRaises(ValueError):summarize([[float('nan')]],key=[1])
        with self.assertRaises(ValueError):summarize([[.5]],key=None)
class AttackTest(unittest.TestCase):
    def test_fixed_attack_and_global_rng_preservation(self):
        import torch
        from saturday_channels import SETTINGS,transform
        x=torch.linspace(0,1,3*64*64).reshape(3,64,64)
        for name in SETTINGS:
            torch.manual_seed(71);before=torch.get_rng_state().clone()
            a=transform(x,name,12);torch.testing.assert_close(torch.get_rng_state(),before)
            b=transform(x,name,12);torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertEqual(a.shape,x.shape);self.assertTrue(torch.isfinite(a).all())
if __name__=='__main__':unittest.main()
