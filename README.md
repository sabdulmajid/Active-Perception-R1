# Active-Perception-R1

Train VLMs to **look before they answer**.

`Active-Perception-R1` is a research project for multimodal RL where the model is rewarded not only for the final answer, but also for *how it inspects the image* (e.g., `<zoom_roi .../>` in `<think>`).

## Why this project exists

Most VLM pipelines still treat perception as one-shot encoding. That fails on tasks where evidence is tiny (insets, labels, OCR regions).

This project addresses that by combining:

- **Outcome reward**: final answer correctness
- **Process reward**: whether zoom/crop actions were valid and evidence-aligned

## What this project does today

- Ships a custom active-perception reward: [src/active_perception_r1/rewards/active_vision_reward.py](src/active_perception_r1/rewards/active_vision_reward.py)
- Parses and validates `<zoom_roi .../>` traces: [src/active_perception_r1/utils/trace_parser.py](src/active_perception_r1/utils/trace_parser.py)
- Simulates evidence-bearing crop observations: [src/active_perception_r1/envs/zoom_simulator.py](src/active_perception_r1/envs/zoom_simulator.py)
- Includes benchmark runner with real model inference: [scripts/benchmark_active_vision.py](scripts/benchmark_active_vision.py)
- Includes GRPO launcher tuned for 2x RTX Pro 6000: [scripts/train_grpo_active_vision.sh](scripts/train_grpo_active_vision.sh)

## What results you can trust (real runs)

These are actual benchmark artifacts in [reports/active_benchmark](reports/active_benchmark).

### Multi-run summary (n=24 per run, 3 runs/model)

| Model | Baseline Acc (mean) | Oracle Crop Acc (mean) | Active Two-Pass Acc (mean) | Active Crop Usage (mean) | Active - Baseline |
|---|---:|---:|---:|---:|---:|
| SmolVLM-256M | 0.9167 | 1.0000 | 0.5556 | 0.0000 | -0.3611 |
| SmolVLM-500M | 0.9306 | 1.0000 | 0.4306 | 0.0417 | -0.5000 |

### Interpretation (plain English)

- **Oracle crops always help** (`1.0`): perception headroom is real.
- **Prompt-only active loop underperforms** today: small models rarely emit valid zoom actions.
- This repo gives you a **measurable bottleneck** (tool-use alignment), not hand-wavy claims.

Reference report: [reports/smoke-2026-03-23.md](reports/smoke-2026-03-23.md)

## Latest perf iteration (simple before/after)

Intervention: `strict_zoom` active strategy in [scripts/benchmark_active_vision.py](scripts/benchmark_active_vision.py)

- What changed: force a valid `<zoom_roi .../>` action first, then answer with reinjected crop evidence.
- Evaluation: same model (`SmolVLM-500M`), same protocol, matched seeds (`31`, `37`), `n=24` each.

| Metric | Before (`default`) | After (`strict_zoom`) | Delta |
|---|---:|---:|---:|
| Active accuracy | 0.3958 | 0.9583 | +0.5625 |
| Active crop usage | 0.0208 | 0.3958 | +0.3750 |
| Active - Baseline | -0.5417 | +0.0208 | +0.5625 |

Artifacts:

- [reports/active_benchmark/benchmark-20260323-221157.md](reports/active_benchmark/benchmark-20260323-221157.md)
- [reports/active_benchmark/benchmark-20260323-221315.md](reports/active_benchmark/benchmark-20260323-221315.md)
- [reports/active_benchmark/benchmark-20260323-221456.md](reports/active_benchmark/benchmark-20260323-221456.md)
- [reports/active_benchmark/benchmark-20260323-221612.md](reports/active_benchmark/benchmark-20260323-221612.md)

## Why someone would choose this

Choose this project if you want to:

- Build **verifiable** active-perception RL for VLMs (not generic chat tuning)
- Debug *why* a model failed visually (bad/no zoom) instead of only seeing final accuracy
- Run a practical workstation setup with reproducible scripts and artifacts
- Start from a scaffold that is honest about current limits and easy to extend

## Example: what “active perception” means

Model trace pattern:

```xml
<think>
I need to inspect the tiny inset.
<zoom_roi x0="0.68" y0="0.05" x1="0.95" y1="0.30" />
Now I can read the value.
</think>
<answer>42</answer>
```

Reward signal behavior:

- Correct answer + targeted zoom over relevant region ⇒ higher score
- Correct answer but no zoom on zoom-required task ⇒ penalized process reward
- Random/tiny/invalid zoom ⇒ penalized process reward

## Quick start

### 1) Validate core logic

```bash
cd /pub7/neel2/active-perception
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

### 2) Run real model benchmark

```bash
cd /pub7/neel2/active-perception
PYTHONPATH=src:.vendor HF_HOME=/pub7/neel2/.hf_home HUGGINGFACE_HUB_CACHE=/pub7/neel2/.cache_hf HF_HUB_DISABLE_XET=1 \
python3 scripts/benchmark_active_vision.py --samples 24 --model-id HuggingFaceTB/SmolVLM-500M-Instruct --output-dir reports/active_benchmark
```

### 3) Run GRPO launcher

```bash
cd /pub7/neel2/active-perception
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct N_RESPONSES=2 TRAIN_BATCH_SIZE=8 VAL_BATCH_SIZE=8 \
./scripts/train_grpo_active_vision.sh
```

## Repository structure

```text
configs/           sample task schema
scripts/           training + benchmark runners
src/               reward, parser, simulator implementations
tests/             unit tests for reward/parser behavior
reports/           benchmark outputs and smoke summaries
```

## Current status and next milestone

Current status:

- Reward + parser + simulator implemented and tested
- Real benchmark harness implemented and producing reproducible reports
- Core finding established: perception headroom exists; zoom policy is the bottleneck

Next milestone:

- Improve tool-use alignment (supervised warm-start + stricter format control)
- Re-run same benchmark protocol and show positive `active_minus_baseline`

## Project intent

This repository is designed to be a **credible MLE research scaffold**: clear hypothesis, measurable protocol, explicit failure modes, reproducible artifacts, and no inflated claims.
