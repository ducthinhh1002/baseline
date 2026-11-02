# Original-method teacher → student benchmark

Implementation follows `baseline_guide.txt`. Start with [METHOD_AUDIT.md](METHOD_AUDIT.md) and [DATA_DISTILLATION_AUDIT.md](DATA_DISTILLATION_AUDIT.md). Audit uncertainties are explicit; CPU checks do not claim reproduced paper results.

**No model weights or large datasets are downloaded, and no local/server GPU experiment is launched by setup.** Original root `campaign.py`, `backends.py`, `metrics.py`, `student.py`, `download.py`, `smoke.py` are historical code for the superseded agg256 experiment. Use the new `benchmark` entry point. `run.sh` now only verifies source and prepares a CPU plan.

## Layout

`benchmark/methods/*/official/`: pinned author source/licenses, copied unchanged.
`benchmark/methods/native.py`: author embedding/verifier integration, no private Structure teacher training.
`benchmark/shared/data/prompts.json`: exact new COCO distillation5000/validation1000/test1000.
`benchmark/shared/student_training/`: wrappers and verbatim historical student trainers.
`benchmark/shared/evaluation/`: method-native metrics and frozen clean-teacher calibration; optional utility.
`benchmark/shared/manipulations/`: Saturday shared transformations, plus clean condition.
`benchmark/configs/`:60configuration grid, source hashes, asset-path example.

## CPU preparation (this task)

```bash
python -m benchmark.cli verify-sources
python -m benchmark.cli plan --work guide_work
python -m unittest discover -s benchmark/tests -v
```

Plan contains no commands that run training or downloads. Empty calibration and unknown teacher-training membership remain explicit, not treated as verified. Add `--calibration independent_prompts.json` and `--teacher-training author_training_manifest.json` when available. Read audit limitations for pretrained methods.

## Operator-only GPU stages

Use method-specific environments: preserve author dependency versions for GaussMarker's original inverse pipeline; do not silently substitute a handwritten inversion when old APIs fail. Install dependencies from each official source and the pinned historical student environment (Torch2.7.1, Diffusers0.35.1, Transformers4.51.3, PEFT0.17.1, Accelerate1.10.1). No GPU compatibility test was run in this revision.

Prepare local artifacts manually. `--artifacts` uses `teachers/stable_signature/{sd2_decoder.pth,extractor.pt}`, `teachers/serum/{checkpoint.pth,backbone.json}` (backbone JSON has local `path`), and operator-provided GaussMarker key/GNR/fuser paths in assets JSON. TrustMark official Q YAML/encoder/decoder must be supplied in its native `official/python/trustmark/models/` directory. Author checkpoints/licenses remain the operator's responsibility. Do not use arbitrary replacement backbone with SERUM.

Example Stable Signature (run only by operator on GPU):

```bash
python -m benchmark.cli generate --execute --method stable_signature --assets assets.json --artifacts /local/artifacts --work work/ss --split train --output data/ss/images/teacher_wm_train
# Copy the same prompts.json to data/ss/prompts.json; generate validation/test analogously.
python -m benchmark.cli student-train --execute --arch sd14 --mode r32 --model /models/sd14 --input data/ss --work students/ss/sd14_r32
python -m benchmark.cli student-generate --execute --arch sd14 --mode r32 --model /models/sd14 --work students/ss/sd14_r32 --output student_test/ss/sd14_r32
python -m benchmark.cli score --execute --method stable_signature --assets assets.json --artifacts /local/artifacts --work work/ss --input student_test/ss/sd14_r32 --attack clean --output scores.json
python -m benchmark.cli evaluate --method stable_signature --input scores.json --output result.json
```

For DiTs use `--arch sana600m` or `pixart512`; mode is full/r32/r16/r4. Workers reuse historical cache/training functions; no watermark loss or marked teacher VAE is used. Validation images remain separate; fixed endpoint20000 is used.

For GaussMarker/SERUM: generate `--split calibration --calibration independent_prompts.json` (clean teacher), score with `--split calibration --attack ATTACK`, then:

```bash
python -m benchmark.cli calibrate --method serum --input clean_scores.json --output frozen_threshold.json
python -m benchmark.cli evaluate --method serum --input student_scores.json --threshold frozen_threshold.json --output result.json
```

Use that SAME per-attack frozen threshold for teacher and every student. Threshold function rejects unsupported methods; evaluate rejects attack mismatch. Native metrics: Stable Signature analytical detection at1e-6; TrustMark raw100-bit BA (ECC OFF); GROW exact16-bit M-ACC; GaussMarker fused TPR@1%FPR plus restored BA; SERUM scalar TPR@1%FPR. No model aggregation, FKR, common bit threshold or student-specific calibration.

Utility helper accepts precomputed reference images plus locally prepared CLIP model, reports FID/KID and explicitly named raw CLIP cosine (not a silently rescaled score). Torch-fidelity may require its Inception weights to be pre-cached on the operator machine. This task does not run utility downloads.
