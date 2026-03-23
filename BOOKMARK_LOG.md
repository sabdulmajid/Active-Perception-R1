# AeroRL — Session Bookmark Log

This file is updated after every meaningful commit so that a new agent can
pick up exactly where work left off.  Read this first at the start of any
session.

---

## Session 1 — 2026-03-23  (AeroRL Phase 1 + Phase 2 MVP)

### What was built

Complete open-source `aerorl/` library committed to this repo.  All code is
in the `aerorl/` top-level package.  The library is importable from the repo
root with `PYTHONPATH=.`.

### Files created

| File | Purpose |
|---|---|
| `aerorl/__init__.py` | Public API: `AeroRLConfig`, `wrap_vlm_for_rl`, all re-exports |
| `aerorl/config.py` | `AeroRLConfig` dataclass with full docstring |
| `aerorl/extensions/__init__.py` | Re-exports `AeroRLSharedKVCache` |
| `aerorl/extensions/ipc_kv_cache.py` | `AeroRLSharedKVCache` zero-copy KV class + IPC fallback |
| `aerorl/extensions/csrc/ipc_kv_cache.cu` | CUDA IPC extension source (builds with `setup.py`) |
| `aerorl/kernels/__init__.py` | Re-exports `grpo_loss`, `gspo_loss`, `cispo_loss` |
| `aerorl/kernels/grpo_loss.py` | Triton GRPO kernel + PyTorch fallback + autograd.Function |
| `aerorl/kernels/gspo_loss.py` | Triton GSPO kernel + PyTorch fallback + autograd.Function |
| `aerorl/kernels/cispo_loss.py` | Triton CISPO kernel + PyTorch fallback + autograd.Function |
| `aerorl/kernels/dapo_filter.py` | DAPO variance filter (Phase 2) |
| `aerorl/utils/__init__.py` | Re-exports all utils |
| `aerorl/utils/vision_mask.py` | `build_vision_mask_*` helpers |
| `aerorl/utils/quant_ref.py` | `QuantisedRefModel`, `BackgroundRefHook`, `gather_log_probs_and_free` |
| `aerorl/utils/processor_utils.py` | `VisionMaskBuilder`, `detect_family` — auto-masks for Qwen2.5-VL, LLaVA-1.6, InternVL2, Phi-3-Vision |
| `aerorl/benchmarks/__init__.py` | Package stub |
| `aerorl/benchmarks/vlm_grpo_benchmark.py` | Full benchmark script with `--dry-run` mode |
| `aerorl/examples/__init__.py` | Package stub |
| `aerorl/examples/basic_usage.py` | End-to-end CPU example (no GPU required) |
| `aerorl/examples/qwen2_5_vl_grpo.py` | Full Qwen2.5-VL GRPO training template |
| `setup.py` | CUDA 12.4+ wheel build config (CUDAExtension + Triton) |
| `tests/test_aerorl.py` | 52 unit tests (16 run without GPU, 36 skipped when torch absent) |

### Verification

```bash
PYTHONPATH=. python3 -m unittest tests/test_aerorl.py -v
# → 52 tests: 16 OK, 36 skipped (PyTorch not installed in CI env)

PYTHONPATH=src python3 -m unittest discover -s tests -v
# → 64 total: 28 OK (12 original + 16 aerorl), 36 skipped
```

### Test command for GPU environment

```bash
# Install deps first (on a machine with CUDA 12.4+):
pip install -e ".[all]"

# Run all tests (all 64 should pass, 0 skipped):
PYTHONPATH=. python3 -m unittest discover -s tests -v

# Run the end-to-end example:
PYTHONPATH=. python3 -m aerorl.examples.basic_usage

# Run the benchmark (dry-run, CPU):
python3 -m aerorl.benchmarks.vlm_grpo_benchmark --dry-run

# Run the benchmark (GPU, real performance):
python3 -m aerorl.benchmarks.vlm_grpo_benchmark \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --group-sizes 8 16 \
    --seq-lengths 4096 8192 \
    --device cuda:0 \
    --output-json results/aerorl_benchmark.json

# Run the Qwen2.5-VL GRPO example (dry-run):
python3 -m aerorl.examples.qwen2_5_vl_grpo --dry-run
```

### What Phase 1 delivers

- **CUDA IPC zero-copy KV sharing** (`AeroRLSharedKVCache`): exports CUDA IPC
  handles from vLLM rollout; imports as zero-copy `Tensor` views in training.
  Falls back to reference-counted tensor storage when the compiled C++/CUDA
  extension (`aerorl_ipc_ext`) is absent.

- **Three Triton loss kernels** with full `torch.autograd.Function` backward:
  - `grpo_loss` — token-level clip + reverse-KL, vision mask, seq_lengths
  - `gspo_loss` — group-sparse selection (top-k by |advantage|) + GRPO
  - `cispo_loss` — sequence-level clipped importance-sampled PO

- **Quantised reference model** (`QuantisedRefModel`): INT8 via bitsandbytes
  or torchao; frees logits immediately after log-prob gather.

- **Background ref hook** (`BackgroundRefHook`): CUDA-stream overlap.

### What Phase 2 delivers (integrated)

- **DAPO variance filter** (`aerorl/kernels/dapo_filter.py`)
- **Auto vision masking for 4 VLM families** (`VisionMaskBuilder`):
  Qwen2.5-VL, LLaVA-1.6, InternVL2, Phi-3-Vision

### Remaining work for next session

- [ ] Build and test the CUDA IPC extension on a CUDA 12.4 machine:
  `python setup.py build_ext --inplace`
- [ ] Run the full benchmark on RTX PRO 6000 and commit results to
  `results/aerorl_benchmark.json` + `results/aerorl_benchmark.md`
- [ ] Integrate `AeroRLSharedKVCache` with vLLM's `PagedAttention` KV store
- [ ] Add verl integration wrapper calling `grpo_loss`/`gspo_loss` from
  verl's GRPO trainer
- [ ] FP8 quantisation path in `quant_ref.py` via `torchao` FP8 API
- [ ] Write `results/aerorl_benchmark.md` with real GPU numbers

---

## How to continue in a new session

1. Read `BOOKMARK_LOG.md` (this file) and `tasks/todo.md`.
2. Check `AGENTS.md` for coding conventions.
3. Run `PYTHONPATH=. python3 -m unittest discover -s tests -v` — baseline.
4. Install deps on GPU: `pip install -e ".[all]"`.
5. Pick up "Remaining work" items above.
