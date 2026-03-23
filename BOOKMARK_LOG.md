# BOOKMARK_LOG

## 2026-03-23 — Session: Reward API hardening for verl runtime

### Summary
- Hardened reward entrypoint in `src/active_perception_r1/rewards/active_vision_reward.py` so `compute_score` safely handles list-shaped inputs often seen in rollout/trainer plumbing.
- Added coercion helpers:
  - `_coerce_string`
  - `_coerce_dict`
  - `_coerce_aliases`
- Kept reward math unchanged; this is compatibility hardening, not reward-policy redesign.

### Tests Added/Updated
- Updated `tests/test_active_vision_reward.py` with:
  - `test_handles_list_shaped_verl_payload_fields`
  - `test_accepts_scalar_answer_aliases`
- Validation command and result:
  - `PYTHONPATH=src python3 -m unittest discover -s tests -v`
  - `OK (9 tests)`

### Why this matters
- Prevents brittle failures when external pipelines pass batched/list wrappers into custom reward hooks.
- Makes long-running GRPO jobs safer to resume and less likely to crash due to shape/type mismatches.

### Next Actions for any follow-up agent
1. Push this commit if not already pushed.
2. Install `verl` + `vllm` + `torch` in runtime env.
3. Run a smoke launch with small settings (3B model, `N_RESPONSES=2`) from `scripts/train_grpo_active_vision.sh`.
4. If launch fails, capture stack trace and patch adapter/wrapper code without changing reward semantics.

### Quick resume commands
```bash
cd /pub7/neel2/active-perception
git status
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## 2026-03-23 — Session: Actual model benchmarking (smoke)

### What was added
- New runner: `scripts/benchmark_active_vision.py`
  - Builds a synthetic inset-reading benchmark set
  - Runs 3 protocols per sample:
    - `full_image` baseline
    - `oracle_crop` (upper bound on perception headroom)
    - `active_two_pass` (model emits optional `<zoom_roi .../>`, then second pass with crop)
  - Writes machine-readable + markdown reports into `reports/active_benchmark/`

### Actual model impact (n=24 each)
- `HuggingFaceTB/SmolVLM-256M-Instruct`
  - baseline: `0.8750`
  - oracle_crop: `1.0000` (delta `+0.1250`)
  - active_two_pass: `0.5000` (delta `-0.3750`)
  - active crop usage: `0.0000`
- `HuggingFaceTB/SmolVLM-500M-Instruct`
  - baseline: `0.8333`
  - oracle_crop: `1.0000` (delta `+0.1667`)
  - active_two_pass: `0.5417` (delta `-0.2917`)
  - active crop usage: `0.0417`

### Interpretation
- There is measurable upside from better perception policy (`oracle_crop > baseline`).
- Current prompt-only active loop is not enough for these small models; they rarely emit valid zoom actions.
- This is now measured, not assumed.

### Artifacts to inspect
- `reports/smoke-2026-03-23.md`
- `reports/active_benchmark/benchmark-20260323-110341.{json,md}`
- `reports/active_benchmark/benchmark-20260323-110530.{json,md}`

### Runtime notes
- Global env on `/pub3` is disk-full; local deps were installed under `.vendor`.
- Use these env vars for reruns:
  - `PYTHONPATH=/pub7/neel2/active-perception/src:/pub7/neel2/active-perception/.vendor`
  - `HF_HOME=/pub7/neel2/.hf_home`
  - `HUGGINGFACE_HUB_CACHE=/pub7/neel2/.cache_hf`
  - `HF_HUB_DISABLE_XET=1`

## 2026-03-23 — Session: Live reinjection + final cleanup

### What was implemented
- Added a real iterative live-reinjection loop:
  - `src/active_perception_r1/sim/live_reinjection.py`
  - `src/active_perception_r1/sim/__init__.py`
- Benchmark runner now executes active mode via live reinjection:
  - `scripts/benchmark_active_vision.py`
- Added tests for reinjection behavior:
  - `tests/test_live_reinjection.py`
- Hardened launcher knobs for practical workstation execution:
  - `scripts/train_grpo_active_vision.sh`

### Validation
- `PYTHONPATH=src python3 -m unittest discover -s tests -v` => `11/11` passing.

### Artifact curation
- Added benchmark runs from additional n=24 sweeps under `reports/active_benchmark/`.
- Updated `reports/smoke-2026-03-23.md` with multi-run summary.
- Added ignore rules for runtime trash:
  - `.vendor_train/`, `.tmp/`, `reports/grpo_smoke_run*.log`.

### Next agent note
- Core active-perception loop + measurable benchmarking are now implemented and committed.
- Remaining improvements are quality/performance work (e.g., better tool-use alignment), not missing project skeleton.

## 2026-03-23 — Session: Real perf iteration (strict zoom strategy)

### Goal
- Improve active-mode performance in a measurable, simple-to-explain way.

### Change implemented
- Added `--active-strategy` in `scripts/benchmark_active_vision.py` with:
  - `default`
  - `strict_zoom` (force valid `<zoom_roi .../>` output before answer step)

### Measured impact (SmolVLM-500M, matched seeds 31/37, n=24 each)
- Active accuracy: `0.3958 -> 0.9583` (`+0.5625`)
- Active crop usage: `0.0208 -> 0.3958` (`+0.3750`)
- Active minus baseline: `-0.5417 -> +0.0208` (`+0.5625`)

### Artifacts
- `reports/perf-iteration-2026-03-23.md`
- `reports/active_benchmark/benchmark-20260323-221157.{json,md}`
- `reports/active_benchmark/benchmark-20260323-221315.{json,md}`
- `reports/active_benchmark/benchmark-20260323-221456.{json,md}`
- `reports/active_benchmark/benchmark-20260323-221612.{json,md}`
