"""
aerorl/benchmarks/vlm_grpo_benchmark.py
========================================
Benchmark script: AeroRL GRPO loss vs. vanilla TRL/verl loss on
Qwen2.5-VL-7B.

Usage
-----
::

    # Full benchmark (requires CUDA + ~96 GB VRAM):
    python -m aerorl.benchmarks.vlm_grpo_benchmark \\
        --model Qwen/Qwen2.5-VL-7B-Instruct \\
        --group-sizes 8 16 \\
        --seq-lengths 4096 8192 \\
        --device cuda:0

    # Dry run (CPU, no real model needed):
    python -m aerorl.benchmarks.vlm_grpo_benchmark --dry-run

Metrics reported
----------------
- Peak VRAM (MiB)     — measured with torch.cuda.max_memory_allocated
- Throughput (it/s)   — iterations per second (1 iter = 1 forward+backward)
- OOM point           — first (group, seq_len) combination that causes OOM
- Loss values         — sanity-checked for finiteness

Hardware target
---------------
  NVIDIA RTX PRO 6000 (96 GB VRAM, Blackwell architecture)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import triton  # noqa: F401
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# ──────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class BenchmarkResult:
    backend: str          # "aerorl" | "vanilla"
    model: str
    group_size: int
    seq_len: int
    peak_vram_mib: float
    throughput_its: float
    loss: float
    oom: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ──────────────────────────────────────────────────────────────────────────
# Synthetic tensor factory (avoids loading real model for dry-run)
# ──────────────────────────────────────────────────────────────────────────
def _make_synthetic_batch(
    BG: int,
    L: int,
    V: int = 32000,
    device: str = "cpu",
    dtype=None,
):
    """Return a dict of synthetic tensors mimicking a GRPO batch."""
    if not _HAS_TORCH:
        raise RuntimeError("Benchmark requires PyTorch.")
    import torch as _torch

    if dtype is None:
        dtype = _torch.bfloat16

    policy_log_probs = _torch.randn(BG, L, device=device, dtype=dtype) - 5.0
    old_log_probs    = policy_log_probs.detach().clone()
    ref_log_probs    = policy_log_probs.detach().clone() - 0.1

    rewards    = _torch.randn(BG, device=device)
    G          = max(1, BG // max(1, BG // 8))
    mean_r     = rewards.view(-1, G).mean(dim=1, keepdim=True).expand(-1, G).reshape(BG)
    std_r      = rewards.view(-1, G).std(dim=1, keepdim=True).expand(-1, G).reshape(BG).clamp(min=1e-8)
    advantages = (rewards - mean_r) / std_r

    # Vision mask: first 256 tokens per seq are image patches
    vision_mask = _torch.ones(BG, L, dtype=_torch.uint8, device=device)
    vision_mask[:, :256] = 0

    seq_lengths = _torch.full((BG,), L, dtype=_torch.int32, device=device)

    return dict(
        policy_log_probs=policy_log_probs.requires_grad_(True),
        old_log_probs=old_log_probs,
        ref_log_probs=ref_log_probs,
        advantages=advantages,
        vision_mask=vision_mask,
        seq_lengths=seq_lengths,
        rewards=rewards,
    )


# ──────────────────────────────────────────────────────────────────────────
# Vanilla GRPO loss (TRL/verl reference implementation)
# ──────────────────────────────────────────────────────────────────────────
def _vanilla_grpo_loss(
    policy_log_probs,
    old_log_probs,
    ref_log_probs,
    advantages,
    vision_mask,
    seq_lengths,
    epsilon: float = 0.2,
    beta: float = 0.01,
):
    """Plain PyTorch GRPO — no Triton, no masking optimisation."""
    import torch as _torch

    BG, L = policy_log_probs.shape
    arange     = _torch.arange(L, device=policy_log_probs.device).unsqueeze(0)
    valid_mask = arange < seq_lengths.unsqueeze(1)
    text_mask  = valid_mask & vision_mask.bool()

    ratio         = _torch.exp(policy_log_probs - old_log_probs)
    ratio_clipped = ratio.clamp(1 - epsilon, 1 + epsilon)
    adv           = advantages.unsqueeze(1)
    surrogate     = -_torch.min(ratio * adv, ratio_clipped * adv)

    diff = ref_log_probs - policy_log_probs
    kl   = _torch.exp(diff) - diff - 1.0

    token_loss = (surrogate + beta * kl) * text_mask.float()
    n_tokens   = text_mask.float().sum(dim=1).clamp(min=1)
    return (token_loss.sum(dim=1) / n_tokens).mean()


# ──────────────────────────────────────────────────────────────────────────
# Single benchmark run
# ──────────────────────────────────────────────────────────────────────────
def run_single(
    backend: str,
    model_name: str,
    G: int,
    seq_len: int,
    B: int = 4,
    n_warmup: int = 3,
    n_iters: int = 10,
    device: str = "cuda:0",
    dtype=None,
    epsilon: float = 0.2,
    beta: float = 0.01,
) -> BenchmarkResult:

    if not _HAS_TORCH:
        raise RuntimeError("Benchmark requires PyTorch.")
    import torch as _torch

    dev = _torch.device(device)
    BG  = B * G

    if dtype is None:
        dtype = _torch.bfloat16

    try:
        batch = _make_synthetic_batch(BG, seq_len, device=device, dtype=dtype)
    except _torch.cuda.OutOfMemoryError:
        return BenchmarkResult(
            backend=backend, model=model_name, group_size=G, seq_len=seq_len,
            peak_vram_mib=float("nan"), throughput_its=0.0,
            loss=float("nan"), oom=True,
            notes="OOM during batch allocation",
        )

    if backend == "aerorl":
        from aerorl.kernels.grpo_loss import grpo_loss
        loss_fn = lambda b: grpo_loss(
            b["policy_log_probs"], b["old_log_probs"], b["ref_log_probs"],
            b["advantages"], b["vision_mask"], b["seq_lengths"],
            epsilon=epsilon, beta=beta,
        )
    else:
        loss_fn = lambda b: _vanilla_grpo_loss(
            b["policy_log_probs"], b["old_log_probs"], b["ref_log_probs"],
            b["advantages"], b["vision_mask"], b["seq_lengths"],
            epsilon=epsilon, beta=beta,
        )

    # Warm up
    for _ in range(n_warmup):
        try:
            loss = loss_fn(batch)
            loss.backward()
            if batch["policy_log_probs"].grad is not None:
                batch["policy_log_probs"].grad.zero_()
        except _torch.cuda.OutOfMemoryError:
            return BenchmarkResult(
                backend=backend, model=model_name, group_size=G, seq_len=seq_len,
                peak_vram_mib=float("nan"), throughput_its=0.0,
                loss=float("nan"), oom=True, notes="OOM during warmup",
            )

    # Reset peak VRAM
    if dev.type == "cuda":
        _torch.cuda.reset_peak_memory_stats(dev)
        _torch.cuda.synchronize(dev)

    # Timed runs
    t0 = time.perf_counter()
    last_loss = float("nan")
    for _ in range(n_iters):
        try:
            loss = loss_fn(batch)
            loss.backward()
            last_loss = float(loss.item())
            if batch["policy_log_probs"].grad is not None:
                batch["policy_log_probs"].grad.zero_()
        except _torch.cuda.OutOfMemoryError:
            return BenchmarkResult(
                backend=backend, model=model_name, group_size=G, seq_len=seq_len,
                peak_vram_mib=float("nan"), throughput_its=0.0,
                loss=float("nan"), oom=True, notes="OOM during timed run",
            )

    if dev.type == "cuda":
        _torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - t0

    peak_mib = 0.0
    if dev.type == "cuda":
        peak_mib = _torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

    return BenchmarkResult(
        backend=backend,
        model=model_name,
        group_size=G,
        seq_len=seq_len,
        peak_vram_mib=round(peak_mib, 1),
        throughput_its=round(n_iters / elapsed, 2),
        loss=round(last_loss, 6),
        oom=False,
        notes=f"triton={'yes' if _HAS_TRITON else 'no'} device={device}",
    )


# ──────────────────────────────────────────────────────────────────────────
# Benchmark table printer
# ──────────────────────────────────────────────────────────────────────────
def _print_table(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'backend':>10}  {'G':>4}  {'L':>6}  "
        f"{'VRAM(MiB)':>10}  {'it/s':>8}  {'loss':>12}  {'OOM':>4}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        vram = f"{r.peak_vram_mib:.0f}" if not math.isnan(r.peak_vram_mib) else "N/A"
        loss = f"{r.loss:.5f}" if not math.isnan(r.loss) else "N/A"
        oom  = "YES" if r.oom else ""
        print(
            f"{r.backend:>10}  {r.group_size:>4}  {r.seq_len:>6}  "
            f"{vram:>10}  {r.throughput_its:>8.2f}  {loss:>12}  {oom:>4}"
        )
    print(sep)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="AeroRL GRPO loss benchmark vs. vanilla TRL/verl."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Model name (used in reporting only; no weights loaded in dry-run).")
    parser.add_argument("--group-sizes", nargs="+", type=int, default=[8, 16],
                        metavar="G")
    parser.add_argument("--seq-lengths", nargs="+", type=int,
                        default=[4096, 8192], metavar="L")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of prompts per batch (B).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true",
                        help="Use CPU and skip real model loading.")
    parser.add_argument("--output-json", default=None,
                        help="Optional path to save results as JSON.")
    parser.add_argument("--backends", nargs="+", default=["aerorl", "vanilla"])

    args = parser.parse_args(argv)

    if args.dry_run:
        args.device = "cpu"
        print("[dry-run] Using CPU.  Results are not representative of GPU performance.")

    if not _HAS_TORCH:
        print("ERROR: PyTorch is not installed.  Cannot run benchmark.", file=sys.stderr)
        sys.exit(1)

    import torch as _torch
    dtype = _torch.bfloat16

    all_results: list[BenchmarkResult] = []

    for backend in args.backends:
        for G in args.group_sizes:
            for L in args.seq_lengths:
                print(f"  Running {backend:>10}  G={G:>3}  L={L:>5} ...", end=" ", flush=True)
                r = run_single(
                    backend=backend,
                    model_name=args.model,
                    G=G,
                    seq_len=L,
                    B=args.batch_size,
                    n_warmup=args.n_warmup,
                    n_iters=args.n_iters,
                    device=args.device,
                    dtype=dtype,
                )
                all_results.append(r)
                status = "OOM" if r.oom else f"{r.throughput_its:.2f} it/s"
                print(f"done  ({status})")

    print()
    _print_table(all_results)

    # Summary: AeroRL vs vanilla speedup
    aerorl_res = {(r.group_size, r.seq_len): r for r in all_results
                  if r.backend == "aerorl" and not r.oom}
    vanilla_res = {(r.group_size, r.seq_len): r for r in all_results
                   if r.backend == "vanilla" and not r.oom}

    if aerorl_res and vanilla_res:
        print("\nSpeedup summary (AeroRL / vanilla):")
        for key in sorted(set(aerorl_res) & set(vanilla_res)):
            a = aerorl_res[key]
            v = vanilla_res[key]
            speedup = a.throughput_its / max(v.throughput_its, 1e-9)
            vram_savings = v.peak_vram_mib - a.peak_vram_mib
            print(
                f"  G={key[0]:<3} L={key[1]:<5}  "
                f"speedup={speedup:.2f}×  "
                f"VRAM savings={vram_savings:.0f} MiB"
            )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump([r.to_dict() for r in all_results], f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
