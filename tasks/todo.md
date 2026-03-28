# Active-Perception-R1 Task Plan

## Implementation Slice — 2026-03-28 (Runtime Header Fallback + Production Validation)

- [x] Add a deterministic Python dev-header fallback so Triton/vLLM helper compilation does not depend on broken system includes.
- [x] Integrate the header fallback into the GRPO launcher and fail fast in preflight when `Python.h` is still unavailable.
- [x] Add unit tests for header resolution/preflight behavior and re-run the local verification suite.
- [x] Re-check GPU occupancy immediately before the next smoke-training launch.
- [x] Re-run a constrained multi-turn GRPO smoke test and capture the first end-to-end result or next blocker precisely.
- [x] Re-check GPU occupancy immediately before real-world benchmark runs.
- [ ] Re-run hardened real-data benchmarks and record current impact with the productionized path.
- [x] Update task logs/documentation with results, then commit and push the curated changes.

## Continuation — 2026-03-23 (Session Hardening)

- [x] Identify a concrete implementation slice that improves real verl training robustness.
- [x] Harden reward input handling for list-shaped / mixed payload fields from rollout pipelines.
- [x] Add regression tests for list payloads and scalar alias metadata.
- [x] Re-run local unit tests after edits.
- [x] Write a bookmark handoff log so work survives disconnects.
- [x] Commit and push latest hardening changes.

## Continuation — 2026-03-23 (Model Benchmarking)

- [x] Install minimal local inference dependencies under project storage due `/pub3` disk saturation.
- [x] Implement a real benchmarking script (`scripts/benchmark_active_vision.py`) that runs actual model inference.
- [x] Execute benchmark protocol with `full_image`, `oracle_crop`, and `active_two_pass` on `HuggingFaceTB/SmolVLM-256M-Instruct` (`n=24`).
- [x] Execute benchmark protocol with `full_image`, `oracle_crop`, and `active_two_pass` on `HuggingFaceTB/SmolVLM-500M-Instruct` (`n=24`).
- [x] Write reproducible benchmark summary to `reports/smoke-2026-03-23.md`.
- [x] Commit and push benchmarking artifacts.

## Continuation — 2026-03-23 (Live Reinjection Finalization)

- [x] Implement iterative live image reinjection loop in `src/active_perception_r1/sim/live_reinjection.py`.
- [x] Integrate benchmark active mode with live reinjection (`scripts/benchmark_active_vision.py`).
- [x] Add live reinjection unit tests (`tests/test_live_reinjection.py`).
- [x] Re-run full unit suite (`11/11`).
- [ ] Commit and push curated implementation + artifact updates.

## Continuation — 2026-03-23 (Real-Data DocVQA Expansion)

- [x] Implement real-dataset benchmark harness (`scripts/benchmark_docvqa_suite.py`) for DocVQA model/strategy sweeps.
- [x] Fix DocVQA schema parsing (`query`/`answer` dict handling) and alias-based exact-match scoring.
- [x] Run real-data benchmark entries for 3 models across `default` and `strict_zoom` (6 total entries).
- [x] Complete expanded 5-model sweep (`10` entries total) and refresh consolidated leaderboard.
- [x] Publish consolidated leaderboard in `reports/real_benchmark/leaderboard-2026-03-23.md`.
- [ ] Commit and push latest real-data benchmark + reporting updates.

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

## Review Notes — Continuation 2026-03-23

- Hardened `compute_score` in `src/active_perception_r1/rewards/active_vision_reward.py` to coerce list-shaped `solution_str`, `ground_truth`, and `extra_info` payloads.
- Added alias coercion so `extra_info.answer_aliases` now supports scalar strings as well as list-like values.
- Added two regression tests in `tests/test_active_vision_reward.py`:
	- list-shaped verl payload fields path
	- scalar `answer_aliases` path
- Validation: `PYTHONPATH=src python3 -m unittest discover -s tests -v` now passes `9/9` tests.

## Review Notes — Benchmarking 2026-03-23

- `reports/active_benchmark/benchmark-20260323-110341.md` (`SmolVLM-256M`, `n=24`):
	- baseline `0.8750`
	- oracle crop `1.0000`
	- active two-pass `0.5000`
	- active crop usage `0.0000`
- `reports/active_benchmark/benchmark-20260323-110530.md` (`SmolVLM-500M`, `n=24`):
	- baseline `0.8333`
	- oracle crop `1.0000`
	- active two-pass `0.5417`
	- active crop usage `0.0417`
- Net finding: current prompt-only active loop underperforms baseline on both tested models, while oracle crops show clear headroom for better perception policy.

## Review Notes — Live Reinjection 2026-03-23

- Added `run_live_reinjection_episode` to support iterative zoom->crop->reinject behavior.
- Benchmark active branch now uses the live reinjection episode API instead of a fixed two-step shim.
- Added tests validating both successful reinjection and early-stop behavior.
- Validation status: `PYTHONPATH=src python3 -m unittest discover -s tests -v` passes `11/11` tests.

## Review Notes — Perf Iteration 2026-03-23

- Added benchmark A/B strategy switch (`default` vs `strict_zoom`) in `scripts/benchmark_active_vision.py`.
- Matched-seed evaluation on `SmolVLM-500M` (`seeds 31/37`, `n=24` each) shows:
	- active accuracy `0.3958 -> 0.9583`
	- crop usage `0.0208 -> 0.3958`
	- active-minus-baseline `-0.5417 -> +0.0208`
- Added concise results report at `reports/perf-iteration-2026-03-23.md` and referenced in README.

## Review Notes — Real Data 2026-03-23

- Added `scripts/benchmark_docvqa_suite.py` to run real DocVQA samples (`nielsr/docvqa_1200_examples`) with baseline vs active modes.
- Corrected dataset field handling to use language/text dicts and alias-aware scoring to avoid false negatives.
- Confirmed 10 real benchmark entries (5 models x 2 strategies), consolidated in `reports/real_benchmark/leaderboard-2026-03-23.md`.
- Current aggregate finding: mean `active_minus_baseline` improved from `-0.4000` (`default`) to `-0.0125` (`strict_zoom`).

## Review + Industry Standardization — 2026-03-27

- [x] Audit the repository end-to-end: source, scripts, tests, reports, data schema, and runtime logs.
- [x] Verify local checks that are feasible in-workspace (`unittest`, `bash -n`, GPU inventory, package availability).
- [x] Compare the current launcher and benchmark approach against current official verl, vLLM, and model guidance.
- [x] Fix the benchmark protocol so `strict_zoom` never falls back to full-image answering; report tool failure separately from active-answer accuracy.
- [x] Rewrite README/report claims to distinguish scaffolded behavior from proven multi-turn active perception.
- [x] Rebuild training around verl multi-turn tool execution so a `<zoom_roi .../>` action can trigger crop reinjection during rollout, not only post-hoc reward scoring.
- [x] Introduce an explicit crop/tool environment contract (`tool_config`, observation payloads, raw chat retention, bounded retries, failure states).
- [ ] Replace the toy train/val parquet with a real train/dev mixture for document OCR, chart QA, and grounded region supervision.
- [x] Add benchmark invariants and smoke tests for: no-fallback strict mode, reward payload compatibility, and training startup dependency checks.
- [x] Package the project with pinned training/inference extras and reproducible cache/env setup so `pip install -e .[train]` is enough to launch.
- [ ] Establish a staged model ladder for 2 x RTX Pro 6000:
- [ ] Tier 1 baseline: Qwen2.5-VL-7B or Qwen3-VL-8B
- [ ] Tier 2 stronger research run: Qwen2.5-VL-32B or Qwen3-VL-32B
- [ ] Tier 3 ablations: smaller Qwen/SmolVLM checkpoints for fast reward and prompt-loop experiments

## Implementation Slice — 2026-03-27 (True Multi-Turn Active Perception)

- [x] Implement a repo-local verl agent loop that treats `<zoom_roi .../>` as a real multi-turn action during rollout.
- [x] Track current-view-to-original-image geometry so repeated zooms compose correctly instead of recropping the full image each turn.
- [x] Reinject executed crops back into the conversation as multimodal observations with bounded retries and explicit tool-error states.
- [x] Extend the reward path so executed tool-trace metadata can be scored directly, with parse-only scoring kept as a fallback for benchmarks/tests.
- [x] Wire the GRPO launcher to the new agent loop with clean config files and environment toggles.
- [x] Add unit and fake-server integration tests for the new agent loop, view-bbox composition, and reward/tool-trace handling.
- [x] Re-run local verification (`unittest`, `compileall`, shell validation) and record any remaining runtime blockers.

## Implementation Slice — 2026-03-27 (Benchmark/Runtime Hardening)

- [x] Add a shared benchmark protocol module so strategy behavior is tested independently of heavyweight model loading.
- [x] Make `strict_zoom` fail closed: no full-image fallback, explicit `tool_status`, `tool_retry_count`, and `strict_zoom_satisfied` fields.
- [x] Add a GPU preflight utility and call it before benchmark/train launches; block heavy runs on busy GPUs unless explicitly overridden.
- [x] Add runtime dependency preflight for train launcher (`torch`, `verl`, `vllm`, `pybase64`, `ray`) with actionable error messages.
- [x] Add script-level tests for strict-mode semantics and GPU/dependency preflight parsing.
- [x] Update README install and benchmark sections to reflect the new preflight behavior and honest benchmark interpretation.
- [x] Verify with local unit tests and shell checks; only run GPU experiments if both GPUs are sufficiently idle.
- [x] Commit and push the curated hardening changes.

## Implementation Slice — 2026-03-27 (Honest GPU Validation)

- [x] Fix live reinjection so valid zoom actions take precedence over inline answers in the same model response.
- [x] Re-run the local test suite after the reinjection fix.
- [x] Re-check GPU occupancy immediately before each benchmark or training run.
- [x] Run a hardened synthetic benchmark on a mainstream VLM to verify strict-mode reporting.
- [x] Run a hardened DocVQA benchmark on at least one stronger Qwen model to produce a first honest post-hardening report.
- [x] Attempt a minimal GRPO smoke launch with the local training vendor environment if GPUs remain idle.
- [x] Record results and limitations in the task log, then commit and push.

## Review Notes — Architecture Assessment 2026-03-27

- Local unit tests pass (`11/11`), parser/reward code is readable, and the synthetic/real benchmark scripts make the project easy to inspect.
- The core claim is not yet realized in training: `compute_score` produces `augmented_context`, but the verl launcher never uses a multi-turn/tool-execution path, so rollout remains passive single-turn generation.
- The current `strict_zoom` benchmark path is not actually strict; both benchmark runners fall back to a normal full-image answer if the zoom step fails, which can make active accuracy appear equal to baseline without real tool use.
- Packaging/runtime was scaffold-level at review time: the default Python environment in this workspace could not import `verl` or `vllm`, and the captured GRPO smoke run failed on a missing `pybase64` dependency inside vLLM startup.
- The included parquet training set is only a tiny synthetic sample (`10` train / `2` val); it is useful for plumbing checks, not for learning a robust active-perception policy.
- Best near-term value: harden the benchmark and environment first, then move the RL path onto verl multi-turn tool execution before spending large GPU budget on longer runs.

## Review Notes — Benchmark/Runtime Hardening 2026-03-27

- Added shared benchmark helpers in `src/active_perception_r1/bench/protocol.py` so active-strategy semantics are testable without loading large models.
- `strict_zoom` now fails closed: if the model never emits a valid zoom after bounded retries, the benchmark records tool failure instead of falling back to a full-image answer.
- Added explicit reporting fields across benchmark scripts:
	- `tool_status`
	- `tool_retry_count`
	- `strict_zoom_satisfied`
- Added `src/active_perception_r1/utils/preflight.py` and wired it into benchmark/train entrypoints to:
	- verify GPU occupancy before heavy runs
	- block launches on busy GPUs unless explicitly overridden
	- fail fast on missing runtime dependencies with an install hint
- Added `tests/test_benchmark_protocol.py`; local verification now passes `18/18` tests.
- Additional validation:
	- `bash -n scripts/train_grpo_active_vision.sh` passes
	- `python3 -m compileall src scripts tests` passes
	- direct preflight smoke check correctly refused a 2-GPU training launch while one GPU was busy
- Deliberately did not re-run real benchmarks in this pass because GPU state was not clean enough for a non-contentious experiment:
	- at verification time only `1/2` GPUs was idle
	- the new preflight guard is intended to enforce that discipline

## Review Notes — Honest GPU Validation 2026-03-27

- Re-checked GPU occupancy with `nvidia-smi` immediately before each benchmark and smoke-training launch; both RTX Pro 6000 GPUs were idle during the 2026-03-27 validation runs.
- Fixed `src/active_perception_r1/sim/live_reinjection.py` so a valid `<zoom_roi .../>` takes precedence over an inline `<answer>` in the same response, and later zooms crop the most recent view instead of always recropping the original image.
- Added `tests/test_live_reinjection.py::test_zoom_takes_precedence_over_inline_answer`; local unit verification now passes `23/23`.
- Added `src/active_perception_r1/utils/dataset_schema.py` and wired the train launcher to auto-disable `data.image_key` when parquet prompts already contain structured image content. This prevents the sample active-vision parquet from being filtered down to zero rows by verl's image accounting.
- Hardened synthetic benchmark rerun (`Qwen/Qwen2.5-VL-3B-Instruct`, `strict_zoom`, `n=24`) produced:
	- baseline `1.0000`
	- active `1.0000`
	- oracle crop `0.9167`
	- crop usage `1.0000`
	- strict zoom satisfied `1.0000`
- Hardened real DocVQA rerun (`Qwen/Qwen2.5-VL-3B-Instruct`, `n=16`, seed `29`) produced:
	- `default`: baseline `0.7500`, active `0.1875`, delta `-0.5625`
	- `strict_zoom`: baseline `0.7500`, active `0.1250`, delta `-0.6250`
	- `strict_zoom` crop usage / satisfaction `0.1250`
- Honest interpretation: the old near-parity `strict_zoom` story was benchmark fallback inflation; with fail-closed strict mode, current active prompting is substantially worse than baseline on real DocVQA.
- Minimal GRPO smoke-launch progression:
	- initial local vendor environment lacked `accelerate` and `gguf`
	- after filling those gaps, launcher reached dataset loading and exposed a verl schema mismatch that dropped all rows
	- after dataset-schema resolution, launcher reached worker startup, FSDP init, and Qwen2.5-VL model construction
	- final blocker is now binary compatibility in the local rollout stack: installed `vllm 0.18.0` is outside `verl 0.7.1`'s declared support window and its wheel expects a different `torch` ABI than the currently installed `torch 2.7.0+cu128`
- Follow-up remediation landed in this pass:
	- train dependencies are now pinned to a supported `verl==0.7.1` / `vllm==0.12.0` / `torch==2.9.0` stack with `transformers<5`
	- preflight now performs a deeper `vllm` runtime import and metadata-level compatibility checks so this class of failure is caught before a long Ray/FSDP startup
- direct validation: `PYTHONPATH=src:.vendor_train:.vendor python3 -m active_perception_r1.utils.preflight ...` now fails immediately with a targeted ABI-mismatch error instead of failing deep inside Ray worker initialization

## Review Notes — True Multi-Turn Rollout 2026-03-27

- Added `src/active_perception_r1/rollout/active_perception_agent.py`, a repo-local verl agent loop that executes `<zoom_roi .../>` actions during rollout instead of only parsing them post hoc in the reward function.

## Review Notes — Runtime Bring-Up 2026-03-28

- Added a vendored Python-dev-header resolver and wired it into the train launcher so Triton / torch-inductor helper builds no longer depend on broken system `Python.h` paths.
- Added runtime sanitization for parquet-style multimodal rows and an agent-loop parquet preparer so verl no longer drops or crashes on the repo sample dataset.
- Added a lazy `sitecustomize` patch for `verl.utils.attention_utils` so missing `flash_attn.bert_padding` no longer aborts the training stack at import time.
- Current smoke-run status:
  - validation now completes on the true multi-turn rollout path
  - the first remaining actor-side failure on Qwen2.5-VL was a `CUBLAS_STATUS_EXECUTION_FAILED` in old-log-prob computation
  - disabling the correct engine-level `use_torch_compile` flags changes that failure mode from rotary-embedding crash to rollout-memory pressure, which is progress
- New train-profile hardening landed in this session:
  - model-aware default recommendations now live in `src/active_perception_r1/utils/training_profiles.py`
  - Qwen2.x-VL defaults now prefer `use_fused_kernels=0`, actor/ref compile off, and a lower rollout reservation (`0.45`) unless the user overrides them
- Verification still green after the code changes so far:
  - local unit tests pass
  - `python3 -m compileall src scripts tests` passes
  - `bash -n scripts/train_grpo_active_vision.sh` passes
- Real benchmark reruns are currently blocked again by live GPU contention outside this training job:
  - a separate `vllm serve` process is holding about `37 GiB` on each GPU
  - no additional benchmark/train runs should be started until that process is cleared or the launcher is redirected to uncontended GPUs
- Added `src/active_perception_r1/rollout/zoom_runtime.py` to centralize zoom execution, nested-view bbox composition, crop generation, observation messages, and explicit tool failure states.
- Added `configs/agent_loop/active_perception_zoom_agent.yaml` and wired `scripts/train_grpo_active_vision.sh` to enable multi-turn rollout with the new agent loop.
- Extended `src/active_perception_r1/rewards/active_vision_reward.py` so executed tool traces take precedence over parse-only reconstruction when rollout metadata is available.
- Added new coverage:
	- `tests/test_zoom_runtime.py`
	- `tests/test_active_perception_agent.py`
	- additional executed-trace reward cases in `tests/test_active_vision_reward.py`
- Validation:
	- `PYTHONPATH=src python3 -m unittest discover -s tests -v` passes `30/30`
	- `python3 -m compileall src scripts tests` passes
	- `bash -n scripts/train_grpo_active_vision.sh` passes
- GPU state was checked before train-side validation:
	- GPU 0: `67 MiB`, `0%`
	- GPU 1: `18 MiB`, `0%`
- Train-side blocker in the active shell is now explicit and shallow:
	- `PYTHONPATH=src python3 -m active_perception_r1.utils.preflight ...` fails because the active interpreter does not have the pinned `train` dependencies installed (`verl`, `vllm`, `pybase64`, `ray`, `gguf`)
	- the repo-side wiring is in place, but a fresh `pip install -e .[train]` environment is still required before a real GRPO smoke launch
