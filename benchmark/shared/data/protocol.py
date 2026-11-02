"""CPU-only split/provenance validation. Pretrained training data is not assumed known."""
import hashlib,json
from pathlib import Path

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def text(r):return ' '.join(r['text'].casefold().split())
def validate(prompts, calibration=None, teacher_training=None, training_free=False):
    required={'train':5000,'validation':1000,'test':1000}
    for split,n in required.items():
        if len(prompts[split])!=n:raise ValueError(f'{split} must contain {n} prompts')
    splits={s:prompts[s] for s in required}
    if calibration is not None:
        if not calibration:raise ValueError('Calibration is empty')
        splits['calibration']=calibration
    if teacher_training is not None:splits['teacher_training']=teacher_training
    seen_ids=set();seen_text=set()
    for name,rows in splits.items():
        ids={str(r['image_id']) for r in rows};texts={text(r) for r in rows}
        if seen_ids&ids or seen_text&texts:raise ValueError('Split overlap: '+name)
        if len(ids)!=len(rows):raise ValueError('Repeated image IDs: '+name)
        seen_ids|=ids;seen_text|=texts
    return {'counts':{s:len(v) for s,v in splits.items()},
            'teacher_training_disjoint_verified':training_free or teacher_training is not None,
            'training_free':training_free,
            'limitation':None if training_free or teacher_training is not None else 'Author pretrained training membership unavailable; do not claim verified disjointness'}
