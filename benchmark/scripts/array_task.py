"""Read one prepared job. Never submit jobs or infer model paths."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True)
    for n in ['plan','models','data','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    row=json.loads(a.plan.read_text())['jobs'][a.index];models=json.loads(a.models.read_text())
    cmd=[sys.executable,'-m','benchmark.cli','student-train','--arch',row['arch'],'--mode',row['mode'],
        '--model',models[row['arch']],'--input',str(a.data/row['method']),
        '--work',str(a.output/row['method']/row['arch']/row['mode'])]
    if a.execute:subprocess.run(cmd+['--execute'],check=True)
    else:print(json.dumps(cmd))
if __name__=='__main__':main()
