# Active-Perception-R1

Active-Perception-R1 is a multimodal RL project that trains vision-language models to **actively inspect images before answering**.

Instead of rewarding only final correctness, it also rewards whether the model used meaningful visual actions (for example, valid `<zoom_roi .../>` tool calls that actually cover relevant evidence).

## Why this matters

Most VLM failures on charts, insets, and small-text regions are perception failures, not pure reasoning failures.

This project gives you a practical way to measure and improve that gap with:

- **Outcome reward**: answer correctness
- **Process reward**: visual action quality and evidence alignment

## What you get

- Active-perception reward module: [src/active_perception_r1/rewards/active_vision_reward.py](src/active_perception_r1/rewards/active_vision_reward.py)
- Tool-trace parser for `<zoom_roi .../>`: [src/active_perception_r1/utils/trace_parser.py](src/active_perception_r1/utils/trace_parser.py)
- Crop/evidence simulator: [src/active_perception_r1/envs/zoom_simulator.py](src/active_perception_r1/envs/zoom_simulator.py)
- Live reinjection loop (zoom -> crop -> reinject): [src/active_perception_r1/sim/live_reinjection.py](src/active_perception_r1/sim/live_reinjection.py)
- verl multi-turn agent loop for executable zoom actions during rollout: [src/active_perception_r1/rollout/active_perception_agent.py](src/active_perception_r1/rollout/active_perception_agent.py)
- Real benchmark runners with reproducible reports: [scripts/benchmark_active_vision.py](scripts/benchmark_active_vision.py), [scripts/benchmark_docvqa_suite.py](scripts/benchmark_docvqa_suite.py)
- GRPO training launcher (verl + vLLM settings): [scripts/train_grpo_active_vision.sh](scripts/train_grpo_active_vision.sh)

## What this actually does

Active perception means the model is allowed to **look again before finalizing an answer**.

In this repo, that is implemented as:

1. Model reads the full image and question.
2. Model can emit a zoom tool call (`<zoom_roi .../>`).
3. The system crops that region and re-injects it.
4. Model gives the final answer.
5. Reward checks both final correctness and whether zoom behavior was meaningful.

In short: we do not reward only the final answer. We also reward good visual inspection behavior.

## What the benchmark terms mean

- **Baseline**: one-pass answer from the full image only (no active zoom loop).
- **Active (`default`)**: model may use zoom, but not strictly enforced.
- **Active (`strict_zoom`)**: the benchmark requires a valid zoom action before answering; invalid zoom attempts are now treated as tool failure rather than silently falling back to baseline behavior.
- **Oracle crop**: upper-bound diagnostic where the model is directly shown the answer region crop (for headroom analysis, not a deployable mode).

## Benchmark status

The repository contains real GPU benchmark artifacts, but the older committed `strict_zoom` reports were generated before fail-closed strict-mode hardening. Treat those numbers as exploratory protocol-development results, not canonical evidence of robust active-perception behavior.

The code now distinguishes:

- successful zoom execution
- malformed / invalid zoom attempts
- answered-without-zoom failures in strict mode

Any new benchmark intended for publication should be re-run with the hardened scripts in this commit.

## First post-hardening spot checks (2026-03-27)

These are the first honest reruns after strict-mode fail-closed hardening.

- Synthetic sanity check (`Qwen/Qwen2.5-VL-3B-Instruct`, `strict_zoom`, `n=24`):
  - baseline `1.0000`
  - active `1.0000`
  - oracle crop `0.9167`
  - crop usage `1.0000`
  - strict zoom satisfied `1.0000`
- Real DocVQA spot check (`Qwen/Qwen2.5-VL-3B-Instruct`, `n=16`, seed `29`):
  - `default`: baseline `0.7500`, active `0.1875`, delta `-0.5625`
  - `strict_zoom`: baseline `0.7500`, active `0.1250`, delta `-0.6250`
  - `strict_zoom` crop usage / satisfaction: `0.1250`

Interpretation: once strict mode actually fails closed, the current prompt-level zoom policy is materially worse than baseline on real DocVQA. That was the right forcing function: the repo now includes a true multi-turn verl agent loop for executable zoom actions during rollout, so the next work is environment bring-up and real training on that path rather than more benchmark prompt tuning.

## Exploratory results (real data, pre-hardening protocol)

Dataset: `nielsr/docvqa_1200_examples` (DocVQA test split)

Hardware: dual NVIDIA RTX Pro 6000 (benchmark executed with real model inference on GPU)

Coverage: **10 benchmark entries** = 5 VLMs × 2 active strategies

### Dataset snapshot (what was benchmarked)

- Source dataset: `nielsr/docvqa_1200_examples`
- Total dataset size: 1,200 documents (`train=1000`, `test=200`)
- Fields per example: document image, question, ground-truth answer, answer aliases, OCR words, OCR bounding boxes
- Benchmark slice used here: `test` split, 16 sampled questions per run, seed `29`
- Total evaluated QA instances in this report: 160 (`5 models × 2 strategies × 16 questions`)

### Example questions from the benchmark dataset

- “What the location address of NSDA?” → “1128 SIXTEENTH ST., N. W., WASHINGTON, D. C. 20036”
- “According to budget request summary what is total amount of other expenses?” → “$975.00”
- “Who is ‘presiding’ TRRF GENERAL SESSION (PART 1)?” → “TRRF Vice President”
- “How many nomination committee meetings has Y. C. Deveshwar attended?” → “2”

| Model | Strategy | Baseline Acc | Active Acc | Active-Baseline |
|---|---:|---:|---:|---:|
| SmolVLM-500M | `default` | 0.6250 | 0.5625 | -0.0625 |
| SmolVLM-500M | `strict_zoom` | 0.6250 | 0.6250 | +0.0000 |
| Qwen2.5-VL-3B | `default` | 0.7500 | 0.1875 | -0.5625 |
| Qwen2.5-VL-3B | `strict_zoom` | 0.7500 | 0.7500 | +0.0000 |
| Qwen2.5-VL-7B | `default` | 0.9375 | 0.0000 | -0.9375 |
| Qwen2.5-VL-7B | `strict_zoom` | 0.9375 | 0.9375 | +0.0000 |
| Qwen2-VL-2B | `default` | 0.8750 | 0.6250 | -0.2500 |
| Qwen2-VL-2B | `strict_zoom` | 0.8750 | 0.8750 | +0.0000 |
| LLaVA-1.5-7B | `default` | 0.2500 | 0.0625 | -0.1875 |
| LLaVA-1.5-7B | `strict_zoom` | 0.2500 | 0.1875 | -0.0625 |

### Clear improvement (no jargon)

- With weak zoom control (`default`), active mode underperforms baseline on average.
- With enforced valid zoom (`strict_zoom`), active mode nearly matches baseline on average.
- Mean active-vs-baseline gap improves from `-0.4000` to `-0.0125`.
- Mean active accuracy lift from `strict_zoom` over `default`: `+0.3875`.

Interpretation: these runs showed that prompt-level tool control matters, but they did not yet prove robust active perception because the earlier protocol still allowed benchmark-time fallback behavior in strict mode.

### What this achieved using this library

- The library turns passive one-shot VLM answering into an explicit inspect-then-answer pipeline.
- It measures both **answer correctness** and **quality of visual actions**, so we can diagnose *why* active mode succeeds or fails.
- In this benchmark, adding strict zoom-action control moved active performance from clearly below baseline to almost parity with baseline on average.

### Proof artifacts

- [reports/real_benchmark/leaderboard-2026-03-23.md](reports/real_benchmark/leaderboard-2026-03-23.md)
- [reports/real_benchmark/docvqa-suite-20260323-230000.json](reports/real_benchmark/docvqa-suite-20260323-230000.json)
- [reports/real_benchmark/docvqa-suite-20260323-230000.md](reports/real_benchmark/docvqa-suite-20260323-230000.md)
- [reports/perf-iteration-2026-03-23.md](reports/perf-iteration-2026-03-23.md)
- [reports/smoke-2026-03-23.md](reports/smoke-2026-03-23.md)

## One concrete example

Question (DocVQA style): “What is the invoice total?”

Active-perception trace:

```xml
<think>
Need to inspect the amount block in the lower-right.
<zoom_roi x0="0.68" y0="0.74" x1="0.96" y1="0.95" />
</think>
<answer>$1,284.50</answer>
```

Why this matters: instead of guessing from the full page, the model explicitly zooms where small text usually lives, then answers from that evidence.

## Who should use this

Use this project if you are:

- Building VLM systems where tiny visual evidence matters
- Training with verifiable rewards and want process-level diagnostics
- Comparing perception strategies with reproducible before/after metrics

## Quick start

### 1) Install

```bash
pip install -e .[bench]
```

For training, use a fresh environment. The `train` extra intentionally pins a tested `verl` / `vllm` / `torch` stack because mixing arbitrary CUDA wheel versions is the fastest way to get runtime ABI failures.

### 2) Validate core logic

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

### 3) Run benchmark

```bash
PYTHONPATH=src:.vendor \
python3 scripts/benchmark_active_vision.py \
  --samples 24 \
  --model-id HuggingFaceTB/SmolVLM-500M-Instruct \
  --active-strategy strict_zoom \
  --output-dir reports/active_benchmark
```

The benchmark now performs a GPU occupancy preflight and will refuse to start if the required GPUs are already busy unless you explicitly pass `--allow-busy-gpu`.

### 4) Run GRPO launcher

```bash
pip install -e .[train]

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct \
N_RESPONSES=2 \
TRAIN_BATCH_SIZE=8 \
VAL_BATCH_SIZE=8 \
./scripts/train_grpo_active_vision.sh
```

The train preflight now does a deeper runtime check for `vllm` and will fail early on broken binary stacks such as mismatched `torch` / `vllm` wheels. The supported launcher pins in this repo are currently centered on `verl==0.7.1`, `vllm==0.12.0`, `torch==2.9.0`, `torchaudio==2.9.0`, `torchvision==0.24.0`, and `transformers<5`.

The launcher now enables verl multi-turn rollout with the repo-local agent config at [configs/agent_loop/active_perception_zoom_agent.yaml](configs/agent_loop/active_perception_zoom_agent.yaml). The model emits `<zoom_roi .../>`, the agent loop executes the crop, reinjects the crop plus observation token, and the reward function scores the executed trace directly when that metadata is present.

For Qwen2.x-VL launch stability, the script now applies a conservative training profile by default:

- `USE_FUSED_KERNELS=0`
- `ACTOR_USE_TORCH_COMPILE=0`
- `ACTOR_FSDP_USE_TORCH_COMPILE=0`
- `REF_USE_TORCH_COMPILE=0`
- `REF_FSDP_USE_TORCH_COMPILE=0`
- `ROLLOUT_GPU_MEMORY_UTILIZATION=0.45`

Those defaults are intended to avoid the current Qwen2.x-VL old-log-prob and colocated-rollout failure modes seen on this workstation. If you have a cleaner upstream stack or more headroom, you can still override them explicitly from the shell.

## Scoring behavior

- Correct answer + relevant zoom: positive reward
- Correct answer but poor/invalid zoom behavior: reduced reward
- Wrong answer: low reward regardless of trace

## Repository layout

```text
configs/   agent-loop config and runtime schema
scripts/   training and benchmark runners
src/       reward/parser/simulator/rollout/live-reinjection code
tests/     unit tests
reports/   benchmark artifacts and summaries
```

## Project status

- Core active-perception logic is implemented and tested.
- Benchmarking is implemented with real model runs and saved artifacts.
- Strict-mode benchmarking is now fail-closed instead of silently reverting to baseline behavior.
- End-to-end multi-turn verl tool execution is implemented in the repo-local agent loop and covered by fake-server integration tests.
- A fresh train environment is still required for a real GRPO smoke launch in this shell; the active interpreter currently lacks the pinned `train` dependencies.

This is a production-minded research scaffold: measurable, reproducible, and focused on real model behavior.
