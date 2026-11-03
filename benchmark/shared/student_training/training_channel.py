"""Apply fixed Saturday manipulation before Normalize; no RGB8 roundtrip for tensor channels."""
import os,json
from pathlib import Path

def install():
    attack=os.environ.get('BASELINE_TRAIN_ATTACK')
    if not attack:return
    from torchvision import transforms
    from benchmark.shared.manipulations.saturday import transform,pixel_digest
    lookup=json.loads(Path(os.environ['BASELINE_PIXEL_INDEX']).read_text())
    original=transforms.Normalize
    class ChannelNormalize(original):
        def forward(self,x):
            index=lookup.get(pixel_digest(x))
            if index is None:raise RuntimeError('Training image not in frozen teacher dataset')
            return super().forward(transform(x,attack,index))
    transforms.Normalize=ChannelNormalize
