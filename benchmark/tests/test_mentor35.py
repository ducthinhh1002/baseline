import unittest,json,tempfile,sqlite3
from pathlib import Path
from benchmark.scripts.mentor35 import init,METHODS,ATTACKS

class Mentor35Tests(unittest.TestCase):
    def test_subset_and_no_aggregation(self):
        p=Path(__file__).resolve().parents[1]/'configs/mentor35.json';c=json.loads(p.read_text())
        self.assertEqual(list(c['students']),['sd14'])
        self.assertEqual(list(c['students']['sd14']),['r4'])
        self.assertEqual(len(METHODS)*len(ATTACKS),35)
        self.assertFalse(c['aggregate']);self.assertEqual(c['process_title'],'VLLM::Enginecore')
    def test_exact_queue_count_and_frozen_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);assets=p/'assets.json';assets.write_text('{}')
            cal=p/'calibration.json';cal.write_text(json.dumps([{'image_id':'extra-clean-id','text':'independent clean calibration prompt not in COCO fixture'}]))
            init(p/'run',assets,cal);init(p/'run',assets,cal)
            c=sqlite3.connect(p/'run/queue.sqlite')
            self.assertEqual(c.execute("select count(*) from jobs where kind='student'").fetchone()[0],35)
            self.assertEqual(c.execute("select count(*) from jobs where kind='teacher'").fetchone()[0],5);c.close()
