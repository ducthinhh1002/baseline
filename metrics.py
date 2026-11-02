"""Image/group summaries. Never call a binary detector bit accuracy."""
import numpy as np
from scipy.stats import binom
def bit_threshold(n,alpha):
    for k in range(n+1):
        if binom.sf(k-1,n,.5)<=alpha:return k
    return n+1
def groups(n,size,partitions=100):
    if size==1:return [np.arange(n).reshape(-1,1)]
    if n<size:return []
    rng=np.random.default_rng(20261029+size)
    return [rng.permutation(n)[:n//size*size].reshape(-1,size) for _ in range(partitions)]
def summarize(coordinates,key=None,sizes=(1,4,8,32,64,256),partitions=100,alpha=.0035001833457499743,scalar_thresholds=None):
    x=np.asarray(coordinates,dtype=np.float64)
    if x.ndim==1:x=x[:,None]
    if not np.isfinite(x).all() or not len(x):raise ValueError('Empty/nonfinite decoder output')
    out={'n':len(x),'rows':[],'aggregation':'mean raw decoder coordinates, then sign/threshold','groups_reuse_images_across_partitions':True}
    if key is not None:
        key=np.asarray(key,dtype=bool);assert x.shape[1]==len(key);k=bit_threshold(len(key),alpha)
        out.update(num_bits=len(key),acceptance_min_correct=k,theoretical_fpr=float(binom.sf(k-1,len(key),.5)),fpr_scope='independent fair bit null; not empirical FPR')
    for size in sizes:
        arrays=[x[g].mean(1) for g in groups(len(x),size,partitions)]
        if not arrays:continue
        z=np.concatenate(arrays)
        if key is not None:
            matches=(z>=0)==key;accepted=matches.sum(1)>=k
            row={'bit_accuracy':float(matches.mean()),'exact_full_key_rate':float(matches.all(1).mean())}
        else:
            if scalar_thresholds is None:raise ValueError('Binary detector needs frozen clean-only thresholds')
            threshold=float(scalar_thresholds[str(size)]);accepted=z[:,0]>=threshold;row={'bit_accuracy':None,'scalar_threshold':threshold}
        out['rows'].append({'images_per_group':size,'accepted':int(accepted.sum()),'groups_evaluated':len(accepted),'acceptance':float(accepted.mean()),**row})
    return out
def scalar_calibration(scores,sizes,partitions=100,alpha=.01):
    scores=np.asarray(scores).reshape(-1);thresholds={}
    for size in sizes:
        a=np.concatenate([scores[g].mean(1) for g in groups(len(scores),size,partitions)])
        thresholds[str(size)]=float(np.nextafter(np.quantile(a,1-alpha,method='higher'),np.inf))
    return thresholds
