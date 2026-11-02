# Data and student audit — 2026-11-03

## Data

Exact source is `new_prompt_sd14/source/prompts.json`, copied to `benchmark/shared/data/prompts.json`. SHA256 `5cf63b020893205a1865fa9f238bc81d63d1cd50bdb3d0a0958b007f4cdd84c2`. Train5000/validation1000/test1000; calibration is EMPTY. Image IDs, normalized captions must be disjoint across splits. The historical COCO split was excluded from Structure teacher's old splits, which does NOT prove exclusion from another author's pretrained watermark data. Supply teacher-training manifest for membership checks; GROW is training-free. Unknown public-pretrained membership is exposed in plan output. External OOD manifests can be passed to generation/scoring; no existing non-COCO OOD manifest was verified here. COCO replacement prompts are not themselves an OOD guarantee.

For GaussMarker/SERUM, operator supplies independent nonempty calibration prompts (>=100 recommended;5000 preferred), disjoint from student train/val/test. Generate negatives only with clean SD2.1 teacher, score each shared attack, freeze per-attack threshold. Never derive student thresholds. Dataset output uses `sample_XXXXX.png`, caption `metadata.jsonl`, SHA256 image manifests. Same test prompts for teacher and all students; models independently generate their own outputs. No local images or weights downloaded.

## Existing implementations reused

- SD1.4: `new_prompt_sd14/pipeline/student/{driver,schedule,resume_state,bootstrap}.py`, `harness/student.py::training_command`, pinned Diffusers LoRA/full trainers. Rank32 original20k/batch4/seed0/FP16; r4/r16 only change rank (alpha/r stays1). The original LoRA-only finite-parameter checkpoint guard must be widened for full UNet FT; this one wrapper patch is explicit. Full FT uses the existing full trainer, not a newly authored loop.
- Sana/PixArt: `code_best_pipeline/structure/student32bit_new_prompt_source/cross_arch_jobs.py`, `setup_dit_sweep.py`: actual20k/batch4/seed0, BF16 text/model/latents, FP32 trainables/AdamW/VAE, fixed endpoint; native flow-matching Sana vs epsilon PixArt. The cache/train functions are extracted directly using AST, with path/owner-provenance substitutions and rank changes. Original private carrier/reader evaluation functions are NOT executed.
- PixArt full FT: `prepare_pixart_fullft.py` uses the same training loop, all transformer parameters FP32 under BF16 autocast, VAE/text frozen. New wrapper follows this scope conversion.
- **Missing before integration:** no verified existing Sana FULL-FT implementation was found in the inspected source. This configuration is explicitly new: same existing Sana flow-matching loop, all transformer parameters trainable, FP32 trainables/AdamW + BF16 autocast, frozen VAE/text. It must be GPU-validated before scientific reporting. It is not described as an already reproduced configuration.

Copies and hashes: `benchmark/shared/student_training/reference/`, `benchmark/configs/student_sources.json`. The configuration grid contains60jobs (5methods x3architectures x4adaptations), not105 separate manipulation-trained students. Optional manipulated student training is a separate study factor; default data are clean watermarked teacher images, as guide section4.3. Shared manipulation code is Saturday's exact tensor/RGB8 protocol, not claimed to match every paper's attack severity.

## Selection, validation, resume

Fixed final20000 checkpoint as the inspected historical run. Validation1000 is retained for validation/monitoring, not silently used to choose best detection. SD wrapper preserves epoch data-order/RNG, scaler/optimizer checkpoints1250; DiT original full states1250 contain sampler order/offset and CPU/CUDA/Python/NumPy RNG. No quality or detection early gate. Operator must validate full-FT save/reload and memory on GPU; this task runs only CPU integrity/unit tests. Paths are re-bound per baseline; no marked VAE or private embedding objective enters student training.
