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
- Real benchmark runner with reproducible reports: [scripts/benchmark_active_vision.py](scripts/benchmark_active_vision.py)
- GRPO training launcher (verl + vLLM settings): [scripts/train_grpo_active_vision.sh](scripts/train_grpo_active_vision.sh)

## Proven impact (real runs)

### Baseline multi-run snapshot (n=24, 3 runs/model)

| Model | Baseline Acc | Oracle Crop Acc | Active Acc | Active Crop Usage | Active - Baseline |
|---|---:|---:|---:|---:|---:|
| SmolVLM-256M | 0.9167 | 1.0000 | 0.5556 | 0.0000 | -0.3611 |
| SmolVLM-500M | 0.9306 | 1.0000 | 0.4306 | 0.0417 | -0.5000 |

This established the core bottleneck clearly: **tool-use alignment**.

### Latest performance iteration (matched A/B)

Intervention: strict zoom-action formatting in active mode.

| Metric (SmolVLM-500M, seeds 31/37, n=24 each) | Before (`default`) | After (`strict_zoom`) | Delta |
|---|---:|---:|---:|
| Active accuracy | 0.3958 | 0.9583 | +0.5625 |
| Active crop usage | 0.0208 | 0.3958 | +0.3750 |
| Active - Baseline | -0.5417 | +0.0208 | +0.5625 |

That is a concrete, measured improvement with a simple intervention.

Reports:

- [reports/perf-iteration-2026-03-23.md](reports/perf-iteration-2026-03-23.md)
- [reports/smoke-2026-03-23.md](reports/smoke-2026-03-23.md)

## Who should use this

Use this project if you are:

- Building VLM systems where tiny visual evidence matters
- Training with verifiable rewards and want process-level diagnostics
- Comparing perception strategies with reproducible before/after metrics

## Quick start

### 1) Install

```bash
pip install -e .
```

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

### 4) Run GRPO launcher

```bash
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct \
N_RESPONSES=2 \
TRAIN_BATCH_SIZE=8 \
VAL_BATCH_SIZE=8 \
./scripts/train_grpo_active_vision.sh
```

## Example action trace

```xml
<think>
I need to inspect the inset.
<zoom_roi x0="0.68" y0="0.05" x1="0.95" y1="0.30" />
</think>
<answer>42</answer>
```

Scoring behavior:

- Correct + relevant zoom: positive process reward
- Correct but no zoom on zoom-required task: penalty
- Invalid/random zoom: penalty

## Repository layout

```text
configs/   sample task schema
scripts/   training and benchmark runners
src/       reward/parser/simulator/live-reinjection code
tests/     unit tests
reports/   benchmark artifacts and summaries
```

## Project status

- Core active-perception logic is implemented and tested.
- Benchmarking is implemented with real model runs and saved artifacts.
- Performance improvement path is validated with measured A/B gains.

This is a production-minded research scaffold: measurable, reproducible, and focused on real model behavior.
