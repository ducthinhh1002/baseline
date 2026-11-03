# Mentor35 deployment audit — 2026-11-03

Scope explicitly overrides the guide grid: SD1.4 LoRA r4 only, seven independently manipulated teacher training datasets, five methods =35 students. Use new COCO5000/1000/1000 prompt manifest. Other guide rules remain: author embedding/verifier, ordinary student diffusion loss, frozen clean-teacher thresholds and no invented aggregation. Guide-only prohibition on server runs was superseded by the user's explicit run authorization in this session.

Verifier mapping: Stable Signature48bit statistical detection alpha1e-6; GaussMarker author dual-domain fused detection plus restored256-bit BA; TrustMark100raw-bit BA with ECC OFF; GROW16bit exact message accuracy; SERUM native binary detection at clean-teacher1%FPR. NO aggregate/FKR/ge24of32/common threshold.

Teacher robustness applies each attack to teacher test images, with per-attack clean-teacher calibration for empirical methods. Each student learns one attack on teacher training targets and is evaluated on new unmanipulated outputs using the frozen CLEAN threshold. These two experiment questions are labelled separately; the training attack does not select a student-specific threshold. Flip/rotation15/crop75/drop96/Gaussian0.05/brightness0.7/blur15 are the shared Saturday study settings, not each paper's full original attack table.

The original SD1.4 student driver/trainer is reused. A wrapper inserts the fixed manipulation before Normalize. Drop/crop/Gaussian remain float tensors; they are not saved as a second RGB8 dataset. The lookup verifies each training image against the teacher dataset; RNG is forked by the unchanged Saturday transform. No detection/quality objective enters training. Steps20000, batch4, seed0, fixed final, full optimizer/RNG states1250.

GPU preflight on .21: Stable Signature(48 outputs), TrustMark raw(100), GROW(16), SERUM(scalar), GaussMarker(scalar + payload) all completed native generation/read smoke. This is execution validation, not paper-score reproduction. GaussMarker author code is unchanged; only its old constructor argument binding is wrapped to match Diffusers0.35.1's image_encoder parameter. Author generation/inversion and fused verifier remain direct calls.

Unresolved provenance: public pretrained watermark training membership and exact main-table checkpoint identity remain unverified, as already recorded in METHOD_AUDIT. Do not claim proven teacher-training/distillation dataset disjointness for those public checkpoints. New distillation train/validation/test and calibration split disjointness is checked. This limitation is recorded in deployment config and must accompany resulting reports.

All worker/task/trainer process titles for this deployment are VLLM::Enginecore. No unrelated VLLM process is renamed or stopped. Old five workers/five tasks were stopped; old files/results retained.
