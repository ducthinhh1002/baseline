import unittest
import csv
import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch
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
    def test_gaussmarker_native_256_bits_keep_the_same_nominal_budget(self):
        budget=.0035001833457499743
        k=bit_threshold(256,budget)
        x=np.ones((1000,256));x[:,k:]=-1
        result=summarize(x,[1]*256,sizes=(1,8,32,256),alpha=budget)
        self.assertLessEqual(result['theoretical_fpr'],budget)
        self.assertGreater(k,24)
        self.assertTrue(all(r['acceptance']==1 for r in result['rows']))
        self.assertTrue(all(r['exact_full_key_rate']==0 for r in result['rows']))
    def test_serum_scalar_uses_clean_calibration_without_inventing_bits(self):
        calibration=np.linspace(0,1,5000)
        thresholds=scalar_calibration(calibration,(1,8,32,256),alpha=.0035001833457499743)
        result=summarize(np.ones(1000),scalar_thresholds=thresholds,sizes=(1,8,32,256))
        self.assertNotIn('num_bits',result)
        self.assertTrue(all(r['bit_accuracy'] is None and r['acceptance']==1 for r in result['rows']))
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

class PortableRunTest(unittest.TestCase):
    def test_coco_splits_are_deterministic_disjoint_and_one_caption_per_image(self):
        from prepare_prompts import select_splits
        captions=[dict(id=i*2+j,image_id=i,caption=f'caption {i}/{j}') for i in range(20) for j in range(2)]
        counts={'train':8,'test':4,'calibration':6}
        result=select_splits(captions,counts,17)
        self.assertEqual(result,select_splits(list(reversed(captions)),counts,17))
        ids=[r['image_id'] for split in result.values() for r in split]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertTrue(all(r['text'].endswith('/0') for split in result.values() for r in split))
        with self.assertRaises(ValueError):select_splits(captions,{'train':21},17)
    def test_failed_teacher_does_not_leave_worker_waiting_forever(self):
        import campaign
        with tempfile.TemporaryDirectory() as tmp,patch.object(campaign,'ROOT',Path(tmp)):
            db=campaign.database()
            db.execute('create table tasks(state text)')
            db.executemany('insert into tasks values(?)',[('failed',),('pending',)])
            db.commit();db.close()
            with patch.object(campaign,'claim',return_value=None),patch.object(campaign.time,'sleep') as sleep:
                campaign.worker()
                sleep.assert_not_called()
    def test_detailed_report_does_not_replace_serum_missing_ba_with_zero(self):
        import report
        with tempfile.TemporaryDirectory() as tmp,patch.object(report,'ROOT',Path(tmp)):
            root=Path(tmp)
            for method,key in [('stable_signature',[1]*32),('serum',None)]:
                folder=root/'teachers'/method/'results';folder.mkdir(parents=True)
                result=summarize(np.ones((1000,32 if key else 1)),key,sizes=(1,8,32,256),
                                 scalar_thresholds=None if key else {str(n):.5 for n in [1,8,32,256]})
                (folder/'flip.json').write_text(json.dumps(result))
            with patch.object(report,'config',return_value={'methods':['stable_signature','serum'],'students':[],'attacks':['flip']}):
                report.main()
            rows=json.loads((root/'reports/metrics.json').read_text())['rows']
            self.assertEqual(rows[0]['BA/image'],1.)
            self.assertIsNone(rows[1]['BA/image'])
            self.assertEqual(rows[1]['NK agg32'],1.)
            self.assertNotIn('FID',rows[0])
    def test_launcher_help_and_invalid_arguments_do_not_install_anything(self):
        script=Path(__file__).parent/'run.sh'
        p=subprocess.run(['bash',str(script),'--help'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('105 trainings',p.stdout)
        p=subprocess.run(['bash',str(script),'--gpus','not-a-gpu'],capture_output=True,text=True)
        self.assertEqual(p.returncode,2)
    def test_console_reports_each_completion_once_and_handles_partial_lines(self):
        import io
        from contextlib import redirect_stdout
        from pipeline import completion_lines
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'worker.log'
            path.write_text('step 20 loss=0.1\nDONE model-a\nDONE model-b')
            offsets={};workers=[{'log':str(path)}]
            out=io.StringIO()
            with redirect_stdout(out):
                completion_lines(workers,offsets)
                completion_lines(workers,offsets)
                with path.open('a') as f:f.write('\n')
                completion_lines(workers,offsets)
                completion_lines(workers,offsets)
            self.assertEqual(out.getvalue(),'DONE model-a\nDONE model-b\n')
if __name__=='__main__':unittest.main()
