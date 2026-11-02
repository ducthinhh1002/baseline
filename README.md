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
Restoring those states continues training; FP16 execution across different
GPUs is not guaranteed to reproduce every floating-point bit.
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
GaussMarker uses the pretrained256-bit key/GNR bundle. The default table
now decodes its actual restored256 payload bits; this is not the original fused
score metric. Set `gaussmarker_metric` to `fused_score` only in a separate run
if the published fuser comparison is needed. Source checksums are recorded.
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
SERUM is a scalar detector: it has no native bit key, so its BA is **N/A**,
never invented or zero-filled. Calibrate a threshold for every group size
using5000 separate unwatermarked images at nominal0.350018% FPR, matching
the bit methods' nominal budget (not guaranteeing identical actual FPR).
Its NK column means detection acceptance, not literal key recovery. Optional
GaussMarker `fused_score` mode uses the same scalar protocol. Full JSON
retains exact-key recovery for bit decoders. Partitions reuse images and are
not independent samples. Pending results remain `pending`, never zero.

## Run

Use Linux, an NVIDIA driver, git and Python3.12. A single command creates a
private `.venv`, bootstraps pip even when `ensurepip` is absent, installs
PyTorch2.7.1/torchvision0.22.1 CUDA12.6 and dependencies, prepares disjoint COCO
caption splits if no existing prompt file is available, downloads pinned
sources/models, trains, evaluates, and waits for all jobs to finish:

```bash
./run.sh --gpus 0
```

Default:105 student trainings, each20000 steps. This is the native baseline
benchmark described above, **not** the screenshot's Structure32 rank32 trial.
No FID/KID is calculated. No private Structure teacher checkpoint is needed.
Different native payload lengths are retained; do not force256/48/100 bits
into a24-of32 rule. Native32-bit payloads use the24/32 threshold.

A model completion produces one console line, for example:
`DONE Stable Signature SD1.4 rank4 [flip] | BA/image=... | NK/1=... | NK/8=... | NK/32=... | NK/256=... | BA/256=...`.
Teacher completion prints once after its manipulation panel. Per-step losses,
download progress and upstream diagnostics stay in task logs. The foreground
monitor forwards completion/failure lines only, without replaying old messages.

Before the long queue starts, `run.sh` also runs a bounded GPU preflight for
each selected student architecture using the actual downloaded clean model:
8 training attempts, save/reload of its PEFT adapter and generation of2 images.
These synthetic-fixture checks do not measure watermark transfer. Their logs
are in `work/preflight/`; a failure prevents the long queue from starting.
Successful checks are reused only for the same training code, model manifests,
prompt set, library versions and GPU type. `--prepare-only` stops before these
GPU checks as well as before training. Tiny real Diffusers architecture tests
also check forward/backward and exact adapter-output roundtrips without any
downloads in `test_training.py`.

Trainable LoRA tensors stay FP32. Adapter loading explicitly attaches the PEFT
configuration and sets its state dictionary, then checks all names and tensor
values exactly; it does not guess between similarly named pipeline/model
adapter loaders. Loss, cached latents/embeddings, predictions, gradients and
updated adapter parameters are checked for NaN/Inf. SD1.4 FP16 uses dynamic AMP
scaling starting at1024: an overflow skips the optimizer update and lowers the
scale, without clipping an infinite gradient into NaN.16 consecutive overflows
stop with `failure.json`. BF16/non-AMP nonfinite gradients stop before optimizer
mutation. Gradient clipping accumulates its norm in FP64. Logs distinguish
attempted, applied and skipped updates. These checks limit corruption and catch
API mistakes early; they do not guarantee every hardware/data combination can
finish20000 updates. Use a fresh work directory after changing training code;
existing frozen training identities are deliberately not silently migrated.

```bash
# Put large models/data on the data volume and use GPU1.
BASELINE_WORK=/data/baseline/work BASELINE_CACHE=/data/baseline/cache ./run.sh --gpus 1
# Keep your exact original prompt set rather than creating a new COCO split.
./run.sh --prompts /path/to/coco_prompts.json --gpus 0,1 --workers-per-gpu 1
# Only install/download; do not launch training yet.
./run.sh --prepare-only
# After fixing a failure, resume verified checkpoints and retry failed tasks.
./run.sh --skip-install --retry-failed
```

Use the same work/cache paths and arguments when resuming. A work directory
with a different frozen protocol is rejected; select a new `BASELINE_WORK`
for an older fused-score/1%FPR experiment. A gated model may require `HF_TOKEN`
and accepting its upstream license. If a download fails, the script stops
with its log, never silently substitutes another model. The script installs
Python packages; it does not install the host NVIDIA driver.
Set `TORCH_INDEX_URL` for another official PyTorch wheel variant compatible
with your GPU/driver, and `BASELINE_VENV` to relocate the environment.

The following manual commands remain available:

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
Use5000/1000/5000 disjoint image IDs. The config can reference the old server split. If it is absent, `run.sh`
downloads official COCO2014 captions and creates a deterministic new split,
recording its source archive checksum. This generated split is not claimed
to reproduce the screenshot or exclude an unknown private teacher dataset. Model overrides
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
`run.sh` downloads all selected methods' artifacts before training. Use a
custom config with smaller `methods`/`students`/`attacks` lists to select a
subset, e.g. `./run.sh --config config/subset.json`. The default is one worker
on GPU0. Set additional workers only when the GPU has enough free VRAM.
Results: `work/reports/metrics.csv` and `metrics.json` contain BA/image,
NK/image, NK agg8/32/256 and BA agg256, plus method/model/manipulation,
payload length, threshold, nominal FPR and metric kind. Fractions in these
files range from0 to1; multiply by100 for percentages. SERUM BA cells are
empty/null because it has no bits. The compact
`acceptance_agg256.csv`/`table.json` are also retained. No FID/KID.
Per-model JSON remains authoritative. Teacher download failures are explicit and do not produce
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
