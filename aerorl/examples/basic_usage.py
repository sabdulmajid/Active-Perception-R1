"""
aerorl/examples/basic_usage.py
================================
Minimal end-to-end AeroRL usage example — no real model required.

This script shows the full API surface:
1. Configure AeroRL
2. Build a vision mask from synthetic input_ids
3. Run the GRPO, GSPO, and CISPO losses
4. Use the DAPO variance filter

Run::

    python -m aerorl.examples.basic_usage
"""

from __future__ import annotations

import sys


def main() -> None:
    # ── 1. Configuration ───────────────────────────────────────────────────
    from aerorl import AeroRLConfig

    config = AeroRLConfig(
        zero_copy_kv=True,
        mask_vision_tokens=True,
        loss_type="grpo",
        epsilon=0.2,
        beta=0.01,
        quant_ref_bits=8,
        quant_ref_backend="bitsandbytes",
        dapo_variance_filter=True,
        dapo_min_variance=1e-4,
        vision_processor="qwen2_5_vl",
        max_seq_len=4096,
        dtype="bf16",
    )
    print("Config:", config)

    # ── 2. Check for PyTorch (needed for tensor ops) ───────────────────────
    try:
        import torch
    except ImportError:
        print("PyTorch not installed; skipping tensor demo.")
        return

    BG, L = 4, 128  # small demo batch: B=1, G=4, seq_len=128
    device = "cpu"   # use CUDA device string when GPU is available
    dtype  = torch.float32  # use bfloat16 in production

    # ── 3. Build vision mask ───────────────────────────────────────────────
    from aerorl.utils import VisionMaskBuilder, build_vision_mask_from_input_ids

    # Simulate Qwen2.5-VL input_ids where first 32 tokens are image patches
    input_ids = torch.randint(0, 32000, (BG, L), dtype=torch.long)
    input_ids[:, :32] = 151655  # IMAGE_PAD_ID for Qwen2.5-VL

    builder = VisionMaskBuilder(family="qwen2_5_vl")
    vision_mask = builder.build(
        {"input_ids": input_ids},
        response_start_ids=torch.full((BG,), 64, dtype=torch.long),
    )
    n_text_tokens = vision_mask.sum().item()
    print(f"Vision mask: shape={tuple(vision_mask.shape)}, "
          f"text tokens={n_text_tokens}/{BG*L}")

    # ── 4. Create synthetic log-prob tensors ────────────────────────────────
    torch.manual_seed(42)
    policy_log_probs = torch.randn(BG, L, dtype=dtype) - 5.0
    policy_log_probs.requires_grad_(True)
    old_log_probs = policy_log_probs.detach().clone()
    ref_log_probs = policy_log_probs.detach().clone() - 0.05

    # Group-relative advantages
    G = 4
    rewards    = torch.tensor([1.0, 0.5, -0.5, -1.0], dtype=dtype)
    mean_r     = rewards.mean()
    std_r      = rewards.std(unbiased=False).clamp(min=1e-8)
    advantages = (rewards - mean_r) / std_r

    seq_lengths = torch.full((BG,), L, dtype=torch.int32)

    # ── 5. DAPO variance filter ────────────────────────────────────────────
    from aerorl.kernels.dapo_filter import apply_dapo_filter

    rewards_f, advantages_f, keep_mask = apply_dapo_filter(
        rewards, advantages, G=G, min_variance=config.dapo_min_variance
    )
    n_kept = keep_mask.sum().item()
    print(f"DAPO filter: {n_kept}/{BG} sequences kept "
          f"(reward variance={rewards.var(unbiased=False).item():.4f})")

    # ── 6. GRPO loss ───────────────────────────────────────────────────────
    from aerorl.kernels import grpo_loss

    loss_grpo = grpo_loss(
        policy_log_probs, old_log_probs, ref_log_probs,
        advantages_f, vision_mask, seq_lengths,
        epsilon=config.epsilon, beta=config.beta,
    )
    loss_grpo.backward()
    grad_norm = policy_log_probs.grad.norm().item()
    print(f"GRPO loss = {loss_grpo.item():.6f},  grad norm = {grad_norm:.6f}")
    policy_log_probs.grad.zero_()

    # ── 7. GSPO loss ───────────────────────────────────────────────────────
    from aerorl.kernels import gspo_loss

    loss_gspo = gspo_loss(
        policy_log_probs, old_log_probs, ref_log_probs,
        advantages_f, vision_mask, seq_lengths,
        G=G, top_k_ratio=0.5,
        epsilon=config.epsilon, beta=config.beta,
    )
    loss_gspo.backward()
    grad_norm_g = policy_log_probs.grad.norm().item()
    print(f"GSPO loss = {loss_gspo.item():.6f},  grad norm = {grad_norm_g:.6f}")
    policy_log_probs.grad.zero_()

    # ── 8. CISPO loss ──────────────────────────────────────────────────────
    from aerorl.kernels import cispo_loss

    loss_cispo = cispo_loss(
        policy_log_probs, old_log_probs, ref_log_probs,
        advantages_f, vision_mask, seq_lengths,
        epsilon_seq=config.epsilon, beta=config.beta,
    )
    loss_cispo.backward()
    grad_norm_c = policy_log_probs.grad.norm().item()
    print(f"CISPO loss = {loss_cispo.item():.6f},  grad norm = {grad_norm_c:.6f}")

    # ── 9. Shared KV cache (IPC fallback demo) ─────────────────────────────
    from aerorl.extensions import AeroRLSharedKVCache

    num_layers, num_heads, head_dim = 2, 8, 64
    seq, bsz = 16, 1

    kv_pairs = [
        (
            torch.randn(bsz, num_heads, seq, head_dim),
            torch.randn(bsz, num_heads, seq, head_dim),
        )
        for _ in range(num_layers)
    ]

    cache = AeroRLSharedKVCache.from_kv_pairs(
        kv_pairs, num_heads=num_heads, head_dim=head_dim
    )
    k0, v0 = cache.get_kv_layer(0)
    print(f"KV cache layer 0: k={tuple(k0.shape)}, v={tuple(v0.shape)}")
    print(cache)

    print("\n✓ basic_usage.py completed successfully.")


if __name__ == "__main__":
    main()
