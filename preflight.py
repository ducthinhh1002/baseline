"""Bounded real-base GPU train/reload/generate checks before a long queue."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from common import ROOT,CACHE,REPO,config,read,write,sha


def main(gpu):
    import torch,diffusers,peft
    c=config()
    identity=dict(code={name:sha(REPO/name) for name in ['student.py','smoke.py','training_checks.py']},
        models={arch:sha(CACHE/'models'/f'{arch}.json') for arch in c['students']},
        prompts=sha(ROOT/'prompts.json'),torch=torch.__version__,diffusers=diffusers.__version__,
        peft=peft.__version__,gpu_name=torch.cuda.get_device_name(0))
    token=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    folder=ROOT/'preflight'/token;folder.mkdir(parents=True,exist_ok=True)
    write(folder/'identity.json',identity)
    for arch in c['students']:
        proof=folder/(arch+'.json')
        if proof.exists():continue
        log=folder/(arch+'.log')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,BASELINE_PREFLIGHT_ID=token,
                 BASELINE_PROMPTS=str(ROOT/'prompts.json'))
        with log.open('a') as f:
            result=subprocess.run([sys.executable,str(REPO/'smoke.py'),'student',arch],env=env,
                                  cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'GPU preflight failed for {arch}; full training was not launched. Details: {log}')
        smoke=ROOT.parent/'smoke'/token/'student'/arch/'passed.json'
        check=read(smoke)
        if check['trained']['applied']<1:raise RuntimeError('GPU preflight applied no optimizer updates')
        write(proof,dict(passed=True,smoke=check,log=str(log),scientific_result=False))
        print('PREFLIGHT PASS',arch,'8 updates attempted; adapter reload + image generation passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',default='0');args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    main(args.gpu)
