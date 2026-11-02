# Native watermark baselines: manipulation-to-student transfer

Five methods: Stable Signature, GaussMarker, TrustMark-Q, GROW and SERUM.
Three independently initialized clean students: SD1.4, Sana0.6B512,
PixArt-Alpha512. LoRA rank4, alpha4. There are **105 independent trainings**:
five teachers × three student architectures × seven image manipulations.
This is image-only transfer, not fine-tuning an already-watermarked student.

## Exact experiment question

Each teacher produces 5000 training images and 1000 test images. One fixed
manipulation is applied to each training target before the student's clean VAE
encodes it. Captions are unchanged. Each student then generates 1000 new,
**unmanipulated** test images, decoded by its teacher method's native reader.
Teacher robustness is a separate measurement: manipulate teacher test images
and decode those. These are different experiments; the table deliberately
places them on adjacent rows without claiming the student survived an output
attack. A student's watermark reader is never used as a training loss.

The report has columns `Method`, `Model`, then Flip, Rotation, Crop, Drop,
Gaussian noise, Brightness, Blur, with Teacher and three student rows per method.
Each student cell corresponds to a separate trained adapter. The table contains
20 rows, backed by 105 adapter endpoints and 35 teacher robustness evaluations.

## Saturday protocol (2026-10-31)

| Column | Fixed transformation per training/test image |
|---|---|
| Flip | horizontal flip |
| Rotation | Uniform[-15°,15°], nearest, gray0.5 fill, RGB8 |
| Crop | random square retaining75% area, bilinear/antialias, tensor |
| Drop | retain96% random pixels; others gray0.5; shared RGB mask; tensor |
| Gaussian | sigma0.05 on[0,1], clip[0,1], tensor |
| Brightness | factor0.7, RGB8 |
| Blur | kernel15, sigma Uniform[0.1,2], RGB8 |

Attack code `saturday_channels.py` is copied unchanged from the old experiment.
Seed20261029+image-index for rotation/blur,20261030+index otherwise. Fork only
the CPU RNG, restoring it after transformation. No JPEG or salt-and-pepper.
Method-native reader preprocessing is retained (TrustMark uses PIL/RGB8).

Student budget:20000 attempted updates, batch4, seed0. AdamW1e-4 to7500,
cosine to2e-5 at12500, then constant2e-5. Ordinary epsilon/velocity MSE for
SD1.4/PixArt and native flow-matching MSE for Sana. No quality loss, watermark
loss, best-checkpoint search, or teacher re-training. Final checkpoint fixed.
Full adapter/optimizer/scaler/order/Python/NumPy/CPU/CUDA RNG state every1250.
Save all losses to JSONL. Log actual applied/skipped AMP update counts.
LoRA targets follow the Saturday trainers: SD1.4 attention projections plus
ungrouped Conv2d modules; Sana/PixArt all Linear and ungrouped Conv2d modules.
The actual expanded module names and trainable parameter counts are saved.

Text/latent caching changes I/O, not the objective. Pixel transformations are
cached **before** each clean student's native VAE encoding. PixArt/SD posterior
mean and std are cached, and reparameterized Gaussian samples remain stochastic
at every training batch. The watermark-bearing teacher VAE is never copied to
a student. No code from the private Structure teacher refinement is included.

## Downloads and native methods

`config/upstream.json` pins exact official source revisions. `download.py`
downloads clean student base checkpoints and official teacher artifacts, with
URL/revision/checksum records; never uses random teacher weights as a fallback.
Stable Signature uses the published SD2 decoder and whitened48-bit extractor.
TrustMark uses official Q encoder/decoder with default BCH5 coding.
GaussMarker uses the pretrained256-bit key/GNR bundle and the official repo's
SD2.1 fuser; source checksums are recorded before deserialization.
GROW has no learned watermark checkpoint: use its pinned native progressive
guidance implementation with unchanged defaults. SERUM uses the exact author
epoch50 checkpoint and author SD2.1 cache, verified by SHA256, alpha0.5.

The native methods have different payload lengths and score types. Do not force
all methods into a32-bit decoder. Main table: watermark acceptance after
aggregating256 images; also save all metrics for1/4/8/32/64/256. Bit methods
use their own payload length and the old32-bit test's nominal random-match
budget0.350018%, via the binomial survival function. This is theoretical under
independent fair-bit matches, not an empirically verified FPR; BCH coded bits
are dependent. GROW retains native per-image majority-bit decoding and thus
aggregates hard-bit votes, rather than inventing a new coefficient decoder.
SERUM and fused GaussMarker are scalar detectors: threshold each group size
using separate unwatermarked calibration images at nominal empirical1% FPR.
Their table cells mean detection, **not exact full-key recovery**. Never
compare these thresholds as if they were identical FPR guarantees. Full JSON
retains exact-key recovery for bit decoders. Partitions reuse images and are
not independent samples. Pending results remain `pending`, never zero.

## Run

Use Linux/NVIDIA CUDA and Python3.12. Install torch2.7.1/torchvision0.22.1 for
your CUDA platform, then `pip install -r requirements.txt`.

```bash
export BASELINE_WORK=/var/lib/baseline_manipulation/work
export BASELINE_CACHE=/var/lib/baseline_manipulation/cache
python campaign.py initialize --prompts /path/to/coco_prompts.json
python download.py
python tests.py
python campaign.py launch --gpus 0,1 --workers-per-gpu 2
python campaign.py status
python report.py
```

Prompt JSON has `train`, `test`, `calibration` lists with `text` and `image_id`.
Use5000/1000/5000 disjoint image IDs. Defaults reference the old COCO splits
on the research server; supply `--prompts` on any other machine. Model overrides
can point at already-downloaded exact base snapshots using `--overrides` with
a JSON object mapping `teacher`, `sd14`, `sana600m`, `pixart512` to local paths.
Weights and data are ignored by git. Do not commit tokens, PEM files or keys.

An optional `trained_student_overrides` maps `method/architecture/attack` to
`{"repo":"user/adapter","revision":"exact_commit"}`. The checkpoint is
automatically downloaded for evaluation; only use it if its provenance matches
this exact image set and training manipulation. By default students start
fresh; there is no pre-existing adapter for a new transfer experiment.

Only `DONE teacher/student` (and failures) go to worker console logs. Detailed
upstream progress stays in task logs. Follow `work/logs/worker_gpu*.log`.
`run.sh --prompts /path/to/coco_prompts.json` downloads bases first and lets
each teacher task fetch its own artifacts, so one large author download does
not delay every other method. Set GPU workers after checking free VRAM.
Results: `work/reports/acceptance_agg256.csv` and `table.json`; per-model JSON
is authoritative. Teacher download failures are explicit and do not produce
fabricated baseline numbers. Once fixes are made, retry failed queue tasks
explicitly; never silently replace a teacher or alter an in-progress identity.
Use `python campaign.py retry --method serum` after correcting a download
failure. The same command without `--method` resets all failed tasks.
Run `python smoke.py native METHOD` and `python smoke.py student ARCH` for
isolated bounded GPU checks. These are execution checks, not benchmark scores.

## Sources

- https://github.com/facebookresearch/stable_signature
- https://github.com/SunnierLee/GaussMarker
- https://huggingface.co/SunnierLeee/GaussMarker-pretrained
- https://github.com/adobe/trustmark
- https://github.com/luopengchen/GROW
- https://github.com/Hubizon/SERUM
- SERUM author artifacts: https://github.com/Hubizon/SERUM/issues/1#issuecomment-4963328577

Native upstream code/licenses stay in the separately downloaded source cache;
this repository does not relicense those third-party implementations.
