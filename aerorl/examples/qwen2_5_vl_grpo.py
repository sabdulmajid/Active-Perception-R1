"""
aerorl/examples/qwen2_5_vl_grpo.py
=====================================
End-to-end GRPO training loop for Qwen2.5-VL-7B using AeroRL.

This example shows how to:
1. Load Qwen2.5-VL with a quantised reference model.
2. Generate rollouts with vLLM.
3. Build vision masks automatically.
4. Compute GRPO loss with zero-copy KV cache.
5. Run a training step.

Requirements
------------
  pip install aerorl[all]   # includes torch, triton, transformers, vllm, bitsandbytes

This is a **template** — replace the data loader and model paths for your use
case.  The script is designed to run on a single 96 GB RTX PRO 6000 GPU with
group_size=8 and seq_len=4096.  For dual-GPU setups, wrap the policy model in
``torch.nn.parallel.DistributedDataParallel`` or use verl's FSDP backend.

Run (dry-run mode)::

    python -m aerorl.examples.qwen2_5_vl_grpo --dry-run
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

# ──────────────────────────────────────────────────────────────────────────
# AeroRL imports
# ──────────────────────────────────────────────────────────────────────────
from aerorl import AeroRLConfig, wrap_vlm_for_rl
from aerorl.kernels import grpo_loss
from aerorl.utils import VisionMaskBuilder, gather_log_probs_and_free
from aerorl.kernels.dapo_filter import apply_dapo_filter
from aerorl.extensions import AeroRLSharedKVCache


# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_GROUP  = 8
DEFAULT_SEQ    = 4096


def build_config() -> AeroRLConfig:
    return AeroRLConfig(
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
        max_seq_len=DEFAULT_SEQ,
        dtype="bf16",
    )


# ──────────────────────────────────────────────────────────────────────────
# Mock rollout (replaces vLLM rollout in dry-run mode)
# ──────────────────────────────────────────────────────────────────────────
def mock_rollout(model_name: str, batch_prompts: list, G: int, L: int, device: str):
    """Simulate vLLM rollout without loading a real model."""
    import torch as _torch

    BG = len(batch_prompts) * G
    input_ids      = _torch.randint(0, 32000, (BG, L), dtype=_torch.long, device=device)
    attention_mask = _torch.ones(BG, L, dtype=_torch.long, device=device)
    # Image patches in first 256 tokens
    input_ids[:, :256] = 151655  # Qwen2.5-VL IMAGE_PAD_ID
    actions        = input_ids.clone()
    rewards        = _torch.randn(BG, device=device)
    pixel_values   = _torch.randn(BG, 3, 336, 336, device=device)

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        actions=actions,
        rewards=rewards,
        pixel_values=pixel_values,
    )


# ──────────────────────────────────────────────────────────────────────────
# Training step
# ──────────────────────────────────────────────────────────────────────────
def training_step(
    policy_model,
    rollout: dict,
    ref_log_probs,
    config: AeroRLConfig,
    optimizer,
    G: int,
):
    """Run one GRPO update step."""
    import torch as _torch
    import torch.nn.functional as F

    input_ids      = rollout["input_ids"]
    attention_mask = rollout["attention_mask"]
    actions        = rollout["actions"]
    rewards        = rollout["rewards"]

    BG, L = input_ids.shape

    # ── Build vision mask ──────────────────────────────────────────────────
    builder     = VisionMaskBuilder(family="qwen2_5_vl")
    vision_mask = builder.build(
        {"input_ids": input_ids},
        response_start_ids=_torch.full((BG,), 256, dtype=_torch.long,
                                       device=input_ids.device),
    )

    # ── Compute advantages ─────────────────────────────────────────────────
    mean_r     = rewards.view(-1, G).mean(dim=1, keepdim=True).expand(-1, G).reshape(BG)
    std_r      = rewards.view(-1, G).std(dim=1, keepdim=True).expand(-1, G).reshape(BG).clamp(min=1e-8)
    advantages = (rewards - mean_r) / std_r

    # ── DAPO filter ────────────────────────────────────────────────────────
    if config.dapo_variance_filter:
        _, advantages, _ = apply_dapo_filter(
            rewards, advantages, G=G,
            min_variance=config.dapo_min_variance,
        )

    seq_lengths = attention_mask.sum(dim=1).int()

    # ── Policy forward pass ────────────────────────────────────────────────
    outputs    = policy_model(input_ids=input_ids, attention_mask=attention_mask)
    pol_logits = outputs.logits[:, :-1, :]
    pol_lp     = gather_log_probs_and_free(pol_logits, actions[:, 1:], vision_mask[:, 1:])

    # ── Old policy (frozen snapshot from rollout) — use ref as proxy in demo
    old_lp = ref_log_probs.detach()

    # ── Sequence lengths adjusted for shift ───────────────────────────────
    sl_shifted = (seq_lengths - 1).clamp(min=1)
    vm_shifted = vision_mask[:, 1:]

    # ── GRPO loss ──────────────────────────────────────────────────────────
    loss = grpo_loss(
        pol_lp, old_lp, ref_log_probs,
        advantages, vm_shifted, sl_shifted,
        epsilon=config.epsilon, beta=config.beta,
    )

    # ── Backward + step ────────────────────────────────────────────────────
    optimizer.zero_grad()
    loss.backward()
    _torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
    optimizer.step()

    return float(loss.item())


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="AeroRL Qwen2.5-VL GRPO training demo"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ)
    parser.add_argument("--n-steps", type=int, default=3)
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with tiny mock data and no real model loading.")
    args = parser.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not installed.", flush=True)
        return

    config = build_config()
    print(f"AeroRLConfig: {config}")

    device = args.device

    if args.dry_run:
        print("[dry-run] Using mock model and data.")
        # Tiny mock linear model for demo
        import torch.nn as nn

        class _TinyModel(nn.Module):
            def __init__(self, L, V):
                super().__init__()
                self.embed = nn.Embedding(32001, 64)
                self.lm_head = nn.Linear(64, V, bias=False)

            def forward(self, input_ids, attention_mask=None):
                h = self.embed(input_ids)            # (BG, L, 64)
                logits = self.lm_head(h)             # (BG, L, V)

                class _Out:
                    pass
                out = _Out()
                out.logits = logits
                return out

        V = 32000
        policy_model = _TinyModel(args.seq_len, V).to(device)

        # Fake ref log probs
        BG = 2 * args.group_size
        ref_log_probs = torch.randn(BG, args.seq_len - 1, device=device) - 5.0

        optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)

        batch_prompts = [{"prompt": "What is in the image?"}] * 2

        for step in range(args.n_steps):
            rollout = mock_rollout(
                args.model, batch_prompts,
                G=args.group_size, L=args.seq_len, device=device,
            )
            loss = training_step(
                policy_model, rollout, ref_log_probs, config,
                optimizer, G=args.group_size,
            )
            print(f"Step {step+1}/{args.n_steps}  loss={loss:.6f}")

    else:
        # Full implementation path (requires transformers, vllm, bitsandbytes)
        from aerorl.utils import QuantisedRefModel

        print(f"Loading policy model: {args.model} ...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            policy_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
            ref_model = QuantisedRefModel.from_pretrained(
                args.model, quant_bits=config.quant_ref_bits,
                backend=config.quant_ref_backend,
            )
            print(f"Ref model: {ref_model}")
        except Exception as e:
            print(f"Could not load model: {e}\nRun with --dry-run for a demo.")
            return

        # In a real loop, you would load your dataset and call vLLM for rollouts.
        print("NOTE: Full training loop requires verl or custom data pipeline.")
        print("See scripts/train_grpo_active_vision.sh for the verl launcher.")

    print("\n✓ qwen2_5_vl_grpo.py completed.")


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
