"""Per-image native metrics only; no invented aggregation or full-key threshold."""
import math
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import binom

def calibrate(clean_scores, *, method, clean_manifest_sha256, output, attack='clean'):
    if method not in ('gaussmarker','serum'):
        raise ValueError('This method does not use empirical calibration')
    x=np.asarray(clean_scores,dtype=float)
    if x.ndim!=1 or len(x)<100 or not np.isfinite(x).all():
        raise ValueError('Need >=100 finite independent clean-teacher scores')
    # Conservative upper-tail order statistic, strict score>threshold handles ties.
    threshold=float(np.sort(x)[math.ceil(.99*len(x))-1])
    value={'method':method,'attack':attack,'threshold':threshold,'comparison':'>','target_fpr':.01,
           'observed_calibration_fpr':float((x>threshold).mean()),'n':len(x),
           'clean_manifest_sha256':clean_manifest_sha256,
           'source':'independent_clean_SD2.1_teacher_only','schema':1}
    p=Path(output)
    if p.exists() and json.loads(p.read_text())!=value:
        raise ValueError('Frozen threshold changed')
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2))
    return value

def evaluate(method, values, key=None, threshold=None, payload_bits=None):
    x=np.asarray(values,dtype=float)
    if x.size==0 or not np.isfinite(x).all():raise ValueError('Empty/nonfinite reader output')
    if method in ('serum','gaussmarker'):
        if threshold is None or threshold['method']!=method or threshold['source']!='independent_clean_SD2.1_teacher_only':
            raise ValueError('Require frozen independent clean-teacher threshold')
        result={'metric':'TPR@1%FPR','value':float((x.reshape(-1)>threshold['threshold']).mean()),
                'threshold_record_sha256':hashlib.sha256(json.dumps(threshold,sort_keys=True).encode()).hexdigest()}
        if method=='gaussmarker':
            bits=np.asarray(payload_bits);k=np.asarray(key)
            if bits.ndim!=2 or bits.shape[1]!=len(k):raise ValueError('Missing restored payload')
            result['bit_accuracy']=float((bits==k).mean())
        return result
    k=np.asarray(key,dtype=bool)
    if x.ndim!=2 or x.shape[1]!=len(k):raise ValueError('Reader/key dimensions differ')
    match=(x>0)==k
    if method=='trustmark':return {'metric':'raw_BA','value':float(match.mean()),'ecc':False}
    if method=='grow':return {'metric':'M-ACC','value':float(match.all(1).mean())}
    if method=='stable_signature':
        if len(k)!=48:raise ValueError('Expected official 48bit decoder')
        p=binom.sf(match.sum(1)-1,48,.5)
        return {'metric':'analytical_detection','value':float((p<1e-6).mean()),'alpha':1e-6,
                'bit_accuracy':float(match.mean()),'null':'independent Bernoulli(0.5), upper-tail bit-match test'}
    raise ValueError(method)
