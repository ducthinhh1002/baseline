"""Create disjoint COCO caption splits when no server-local prompt file exists."""
import argparse
import json
import random
import zipfile
from pathlib import Path
from common import CACHE, ROOT, read, write, freeze, sha, config

URL = 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip'


def select_splits(annotations, counts, seed):
    # One caption per image; a COCO image must never appear in two splits.
    captions = {}
    for row in sorted(annotations, key=lambda r: r['id']):
        captions.setdefault(row['image_id'], row['caption'].strip())
    ids = sorted(captions)
    random.Random(seed).shuffle(ids)
    if sum(counts.values()) > len(ids):
        raise ValueError('Not enough distinct COCO images for the requested splits')
    result, offset = {}, 0
    for name, n in counts.items():
        result[name] = [dict(image_id=i, text=captions[i]) for i in ids[offset:offset+n]]
        offset += n
    return result


def main(config_path, output):
    c = read(config_path)
    counts = dict(train=c['train_images'], test=c['test_images'], calibration=c['calibration_images'])
    identity = dict(source=URL, seed=c['seed'], counts=counts,
                    selection='sorted first caption per image; Python Random(seed) shuffle of sorted IDs')
    proof = output.with_suffix('.source.json')
    if output.exists():
        previous = read(proof)
        if previous['selection'] != identity or previous['prompts_sha256'] != sha(output):
            raise RuntimeError('Existing generated prompts differ; choose another work directory')
        return
    from download import url_file
    archive = CACHE / 'datasets/annotations_trainval2014.zip'
    url_file(URL, archive)
    with zipfile.ZipFile(archive) as z:
        annotations = json.loads(z.read('annotations/captions_train2014.json'))['annotations']
    freeze(output, select_splits(annotations, counts, c['seed']))
    freeze(proof, dict(selection=identity, archive_sha256=sha(archive), prompts_sha256=sha(output)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=Path(__file__).parent / 'config/default.json')
    p.add_argument('--output', type=Path, default=ROOT / 'input_prompts.json')
    a = p.parse_args()
    main(a.config, a.output)
    print(a.output.resolve())
