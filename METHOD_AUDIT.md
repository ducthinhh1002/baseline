# Native method audit — 2026-11-03

This supersedes the previous 105-job/agg256 benchmark. No experiment was run during this revision. Official source files are copied byte-for-byte with licenses; hashes are in `benchmark/configs/official_sources.json`.

| Source | Commit | Official repository |
|---|---|---|
| stable_signature | `c91217c06e953a4eaf270ef9de4a3939bcea3366` | https://github.com/facebookresearch/stable_signature.git |
| GaussMarker | `4ac9bfd4e152a56bd93c2a06a809ef6ff8e73155` | https://github.com/SunnierLee/GaussMarker.git |
| trustmark | `6df5a0a29b51cbec832695fdf02051a1937e37bd` | https://github.com/adobe/trustmark.git |
| GROW | `6aa69a9c5d4a9e75df457fcca8dfc71b64a6b870` | https://github.com/luopengchen/GROW.git |
| SERUM | `02f4929e30cd976c5cfe65a45ef1bc4ac79bdfed` | https://github.com/Hubizon/SERUM.git |

## Stable Signature

[Paper, sections3.1 and5](https://arxiv.org/html/2303.15435v2). `finetune_ldm_decoder.py`, `run_evals.py`, `decoding.ipynb`: use author SD2 decoder and whitened48-bit extractor. Decoder checkpoint conversion into Diffusers changes parameter names, not watermark training. Native RGB/ImageNet preprocessing; upper binomial-tail bit-match test under fair independent bits. Primary detection alpha1e-6; no empirical calibration. Fine-tuning is author VAE-decoder procedure, never Structure/refine code. Pretrained artifacts must be supplied by the operator. Their teacher-training membership is not supplied with public weights; disjointness cannot be asserted from our COCO split alone.

## GaussMarker

[Paper](https://proceedings.mlr.press/v267/li25ae.html). `gaussmarker_gen.py` and `gaussmarker_det.py`: retain spatial Gaussian-Shading/ChaCha payload, frequency injection, author inverse pipeline, GNR and `sd21_cls2.pkl` score fuser. Read256 restored payload bits for BA and use BOTH domains for scalar detection. The earlier handwritten inversion in root `backends.py` is NOT used by this integration. New native adapter calls `get_image_latents`, `forward_diffusion`, `eval_watermark` directly. Conditional GNR input-channel shape must match the supplied key/GNR bundle. Generation50steps/CFG7.5; inversion50/CFG1/empty prompt; 512square. Clean SD2.1 negative scores alone fix per-attack1%FPR thresholds, reused unchanged for teacher/all students. This clean-only freeze is the guide's distillation evaluation protocol; author evaluator otherwise computes ROC over positives and negatives. Native old pipeline may require its own Diffusers environment; compatibility is NOT GPU-tested in this task.

## TrustMark

[Paper](https://collomosse.com/pubs/Bui-ICCV-2025.pdf). Official `trustmark.py`, `model.py`, `datalayer.py`:100 raw bits, `use_ECC=False`, binary input/output, Q encoder/decoder. Main output is raw BA; no detection threshold, no coded-bit binomial test, no FKR. Preserve native resolution/preprocessing. The distributed Q weights are an operational pretrained choice; matching their checkpoint release to the exact paper main-table training run is not proven by package defaults. This is recorded rather than claimed as exact numerical paper reproduction.

## GROW

[Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Luo_GROW_Watermark_Generation_with_Progressive_Guidance_for_Diffusion_Models_CVPR_2026_paper.html). Official README main-results table explicitly states16bits and M-ACC; repository example `OKOK` is32bits. Use `OK` (16ASCII bits), author progressive `generate_with_watermark`, native inversion-free `_read_bits`, exact16/16-message success. No new coefficient verifier, oracle rotation search, or >=24/32 criterion. Native config/README agree on strength0.5, guidance200, start0.5, midband0.2..0.5, DDIM50/CFG7.5. CVF PDF access was403 during this audit: paper-vs-repo hyperparameter agreement beyond the released table/config remains unverified. Training-free, so no teacher watermark-training dataset to overlap.

## SERUM

[Paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/7444ca5ef9b57c70fb2b91b90e9583a0-Paper-Conference.pdf). Native `Watermark.forward`, trained `WatermarkScoreModel`, VAE posterior sampling and `configs/config_sd21.yaml`. Author epoch50 watermark/detector and matching author SD2.1 backbone are required. Initial-noise injection remains native; no student reader is trained. TPR@1%FPR is calibrated once per shared attack on independent clean SD2.1 teacher images; never on student negatives. No bit metric for binary CNN. No synthetic fallback weights. Public checkpoint training membership is unavailable, so strict training/distillation disjointness remains unverified unless operator supplies provenance.

## Explicit unresolved checks before claiming paper reproduction

Public pretrained-data membership, exact TrustMark checkpoint/table identity, GROW PDF-only details, and GaussMarker old-Diffusers compatibility need operator verification. CPU tests do not establish model-level correctness, image quality or paper scores. `plan.json` exposes unknown disjointness. Do not label unverified runs as strict dataset-disjoint reproduction. Student watermark robustness and manipulation-of-training-images are separate experiment factors; neither is silently substituted for the other.
