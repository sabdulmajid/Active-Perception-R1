# Active-Perception-R1 Task Plan

## Continuation — 2026-03-23 (Session Hardening)

- [x] Identify a concrete implementation slice that improves real verl training robustness.
- [x] Harden reward input handling for list-shaped / mixed payload fields from rollout pipelines.
- [x] Add regression tests for list payloads and scalar alias metadata.
- [x] Re-run local unit tests after edits.
- [x] Write a bookmark handoff log so work survives disconnects.
- [ ] Commit and push latest hardening changes.

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
- [ ] Set the GitHub remote, commit changes, and push to `https://github.com/sabdulmajid/Active-Perception-R1.git`.

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

## Review Notes — Continuation 2026-03-23

- Hardened `compute_score` in `src/active_perception_r1/rewards/active_vision_reward.py` to coerce list-shaped `solution_str`, `ground_truth`, and `extra_info` payloads.
- Added alias coercion so `extra_info.answer_aliases` now supports scalar strings as well as list-like values.
- Added two regression tests in `tests/test_active_vision_reward.py`:
	- list-shaped verl payload fields path
	- scalar `answer_aliases` path
- Validation: `PYTHONPATH=src python3 -m unittest discover -s tests -v` now passes `9/9` tests.
