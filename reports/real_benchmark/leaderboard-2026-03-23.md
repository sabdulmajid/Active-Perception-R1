# Real-Dataset Benchmark Leaderboard (2026-03-23)

Dataset: `nielsr/docvqa_1200_examples` (DocVQA)

Configuration:

- split: `test`
- samples per run: `16`
- seed: `29`
- strategies: `default` vs `strict_zoom`

## Per-Run Results (10 benchmark entries)

| Model | Strategy | Baseline Acc | Active Acc | Active-Baseline | Crop Usage |
|---|---:|---:|---:|---:|---:|
| `HuggingFaceTB/SmolVLM-500M-Instruct` | `default` | 0.6250 | 0.5625 | -0.0625 | 0.0000 |
| `HuggingFaceTB/SmolVLM-500M-Instruct` | `strict_zoom` | 0.6250 | 0.6250 | +0.0000 | 0.0000 |
| `Qwen/Qwen2.5-VL-3B-Instruct` | `default` | 0.7500 | 0.1875 | -0.5625 | 0.0625 |
| `Qwen/Qwen2.5-VL-3B-Instruct` | `strict_zoom` | 0.7500 | 0.7500 | +0.0000 | 0.1250 |
| `Qwen/Qwen2.5-VL-7B-Instruct` | `default` | 0.9375 | 0.0000 | -0.9375 | 0.0000 |
| `Qwen/Qwen2.5-VL-7B-Instruct` | `strict_zoom` | 0.9375 | 0.9375 | +0.0000 | 0.0000 |
| `Qwen/Qwen2-VL-2B-Instruct` | `default` | 0.8750 | 0.6250 | -0.2500 | 0.0000 |
| `Qwen/Qwen2-VL-2B-Instruct` | `strict_zoom` | 0.8750 | 0.8750 | +0.0000 | 0.1250 |
| `llava-hf/llava-1.5-7b-hf` | `default` | 0.2500 | 0.0625 | -0.1875 | 0.0000 |
| `llava-hf/llava-1.5-7b-hf` | `strict_zoom` | 0.2500 | 0.1875 | -0.0625 | 0.8750 |

## Impact Summary

- Mean `active_minus_baseline` (default): `-0.4000`
- Mean `active_minus_baseline` (strict_zoom): `-0.0125`
- Mean active accuracy gain (`strict_zoom - default`): `+0.3875`
- Mean crop-usage gain (`strict_zoom - default`): `+0.2125`

## Raw Artifacts

- `reports/real_benchmark/docvqa-suite-20260323-230000.json`
