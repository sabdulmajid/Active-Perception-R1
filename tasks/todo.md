# Active-Perception-R1 Task Plan

## Objectives

- [x] Review project guidance in `AGENTS.md`, `README.md`, and `BACKGROUND_RESEARCH.md`.
- [x] Verify available GPU resources with `nvidia-smi` before implementation.
- [x] Research current verl support for multi-modal GRPO, with emphasis on Qwen2.5-VL / Kimi-VL, vLLM rollout, and dual-GPU constraints.
- [x] Research adjacent primary-source literature on active perception, visual self-verification, zoom/crop tool use, and skeptical failure modes.
- [x] Create a clean repository structure for code, configs, scripts, tests, and task tracking.
- [x] Implement `scripts/train_grpo_active_vision.sh` for a 2x RTX Pro 6000 workstation using verl + vLLM.
- [x] Ensure the training entrypoint includes the requested memory and throughput optimizations:
- [x] `actor_rollout_ref.model.use_remove_padding=True`
- [x] dynamic batch sizing enabled for actor / ref / rollout log-prob passes
- [x] `actor_rollout_ref.ref.fsdp_config.param_offload=True`
- [x] `+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True`
- [x] Scaffold a Python package for active-perception rewards, environment simulation, and parsing utilities.
- [x] Implement a custom reward function that:
- [x] parses `<think>` traces for `<zoom_roi>` calls
- [x] validates ROI coordinates and penalizes malformed / random tool use
- [x] simulates a crop and appends a structured observation token back into context
- [x] combines outcome reward with a visual perception/process reward
- [x] Add unit tests for parser, crop simulation, reward aggregation, and edge cases.
- [x] Rewrite `README.md` with an intuitive setup guide, design rationale, findings, skeptical analysis, and next experiments.
- [x] Verify the scaffold with local tests and shell validation.
- [x] Set the GitHub remote, commit changes, and push to `https://github.com/sabdulmajid/Active-Perception-R1.git`.

## Architecture Notes

- Use verl's FSDP backend and vLLM rollout as the default path because this is officially documented for multimodal GRPO, while more advanced agentic/multi-turn support is still evolving quickly.
- Treat visual tool use as a custom process reward and environment shim first, rather than pretending we already have end-to-end image-token reinjection wired into verl internals.
- Keep the first open-source milestone honest: a strong scaffold, a reproducible reward prototype, and a research plan that distinguishes supported behavior from future work.

## Review Notes

- Verified hardware with `nvidia-smi`: 2 x NVIDIA RTX Pro 6000 GPUs, both idle at start.
- Verified code with `python3 -m unittest discover -s tests -v`: 7 tests passed.
- Verified launcher syntax with `bash -n scripts/train_grpo_active_vision.sh`.
- Verified environment limitation: `torch`, `verl`, and `vllm` are not installed in this workspace yet, so no end-to-end GRPO training run was executed locally.
- Research conclusion: the strongest first milestone is a verifiable crop-selection setup with dense process rewards, not a claim of fully solved general active perception.

## Self-Driving Extension

- [x] Audit the untracked self-driving reward, simulator, scripts, and results to reconstruct the intended experiment.
- [x] Fix correctness issues in the synthetic policy sweep before trusting the checked-in metrics.
- [x] Add automated coverage for self-driving rewards, safety penalties, and synthetic report generation.
- [x] Regenerate the synthetic sweep artifacts with the corrected default configuration.
- [x] Update `README.md` to document the self-driving extension, verification commands, and honest limitations.
- [x] Commit changes and push `main` to `origin`.

## Self-Driving Review Notes

- Fixed a cross-seed scene-ID collision in `src/active_perception_r1/sim/self_driving_lab.py` that previously distorted selection-rate and pairwise-preference metrics by grouping different seeded scenes together.
- Verified `bash -n scripts/train_grpo_self_driving.sh`.
- Verified `PYTHONPATH=src python3 -m unittest discover -s tests -v`: `12/12` tests passed after adding self-driving coverage.
- Verified `PYTHONPATH=src python3 scripts/run_self_driving_policy_sweep.py` with the default configuration, producing `9,000` scene instances and `54,000` policy rollouts.
- Refreshed `results/self_driving_policy_sweep.json` and `results/self_driving_policy_sweep.md` from the corrected pipeline.

## AeroRL Library (Zero-Copy VLM RL)

### Phase 1 — COMPLETED (2026-03-23)
- [x] Create `aerorl/` package structure: kernels/, extensions/, utils/, benchmarks/, examples/
- [x] Implement `AeroRLConfig` dataclass with full validation
- [x] Implement `AeroRLSharedKVCache` + CUDA IPC extension (`ipc_kv_cache.cu`)
- [x] Implement Triton GRPO loss kernel with vision mask + autograd.Function backward
- [x] Implement Triton GSPO loss kernel (group-sparse top-k selection)
- [x] Implement Triton CISPO loss kernel (sequence-level clipped IS PO)
- [x] Implement `QuantisedRefModel` (INT8 via bitsandbytes/torchao) + immediate logit freeing
- [x] Implement `BackgroundRefHook` for CUDA-stream overlap
- [x] Implement vision mask helpers (`build_vision_mask_from_labels/input_ids/auto`)

### Phase 2 — COMPLETED (2026-03-23)
- [x] DAPO variance filter (`aerorl/kernels/dapo_filter.py`)
- [x] Auto vision masking for Qwen2.5-VL, LLaVA-1.6, InternVL2, Phi-3-Vision (`VisionMaskBuilder`)

### Build & Benchmark
- [x] `setup.py` with CUDA 12.4+ wheels (CUDAExtension for IPC, Triton for losses)
- [x] `vlm_grpo_benchmark.py` with `--dry-run` mode (CPU-only)
- [ ] Run benchmark on RTX PRO 6000 and embed results in `results/aerorl_benchmark.json`

### Examples & Tests
- [x] `examples/basic_usage.py` — CPU end-to-end demo
- [x] `examples/qwen2_5_vl_grpo.py` — full Qwen2.5-VL training template
- [x] `tests/test_aerorl.py` — 52 tests (16 run on CPU, 36 need GPU/torch)

### AeroRL Review Notes
- All 64 tests pass: 28 run (12 original + 16 new aerorl), 36 skipped (torch absent)
- `PYTHONPATH=. python3 -m unittest discover -s tests -v` is the verification command
- CUDA IPC extension gracefully degrades to tensor-reference fallback when not compiled
- Triton kernels gracefully degrade to PyTorch fallback when not available
- `BOOKMARK_LOG.md` updated with full session notes and next-session instructions
