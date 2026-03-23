# Perf Iteration Report (2026-03-23)

## Goal

Show a **real, measurable** improvement in active-perception performance with a minimal, understandable change.

## Change tested

- Strategy added in `scripts/benchmark_active_vision.py`: `--active-strategy strict_zoom`
- Behavior:
  1. Ask model to emit only a valid `<zoom_roi .../>`
  2. Reinject crop
  3. Answer from original + crop

## A/B setup

- Model: `HuggingFaceTB/SmolVLM-500M-Instruct`
- Samples per run: `24`
- Matched seeds: `31`, `37`
- Compared strategies:
  - `default`
  - `strict_zoom`

## Raw run references

- Seed 31 default: `reports/active_benchmark/benchmark-20260323-221157.md`
- Seed 31 strict: `reports/active_benchmark/benchmark-20260323-221315.md`
- Seed 37 default: `reports/active_benchmark/benchmark-20260323-221456.md`
- Seed 37 strict: `reports/active_benchmark/benchmark-20260323-221612.md`

## Results (mean across 2 matched seeds)

| Metric | Default | Strict Zoom | Delta |
|---|---:|---:|---:|
| Active accuracy | 0.3958 | 0.9583 | +0.5625 |
| Active crop usage | 0.0208 | 0.3958 | +0.3750 |
| Active - Baseline | -0.5417 | +0.0208 | +0.5625 |

## Plain takeaway

A simple output-format intervention turned active mode from "much worse than baseline" into "slightly better than baseline" on these matched runs, while massively increasing actual zoom usage.
