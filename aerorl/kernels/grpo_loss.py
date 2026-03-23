"""
aerorl/kernels/grpo_loss.py
============================
Fused Triton kernel for Group Relative Policy Optimisation (GRPO) loss.

Inputs (all pre-gathered / pre-computed by a thin host wrapper)
---------------------------------------------------------------
policy_log_probs   : (B*G, L) float  — log π_θ(a_t | s_t) for each token
old_log_probs      : (B*G, L) float  — log π_old(a_t | s_t)
ref_log_probs      : (B*G, L) float  — log π_ref(a_t | s_t)
advantages         : (B*G,)   float  — normalised group-relative advantage
vision_mask        : (B*G, L) uint8  — 1 = text/response token, 0 = skip
seq_lengths        : (B*G,)   int32  — valid token count per sequence

Loss formula (per text token, averaged over valid text tokens)
--------------------------------------------------------------
  ratio     = exp(policy_lp - old_lp)
  clipped   = clamp(ratio, 1-ε, 1+ε)
  surrogate = -min(ratio * A, clipped * A)
  kl        = exp(ref_lp - policy_lp) - (ref_lp - policy_lp) - 1   [approx KL]
  loss_t    = surrogate + β * kl
  loss      = mean over tokens where vision_mask == 1

The Triton kernel processes one row (sequence) per program instance.
Each program is launched with grid = (B*G,).

Autograd integration
--------------------
``GRPOLossFunction`` is a ``torch.autograd.Function`` that calls the forward
kernel and implements the backward pass analytically in Triton (or falls back
to PyTorch autograd for the backward when the Triton backward kernel is not
compiled yet — indicated by ``TRITON_BACKWARD_AVAILABLE``).
"""

from __future__ import annotations

import math
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────
# Optional Triton import
# ──────────────────────────────────────────────────────────────────────────
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

# ──────────────────────────────────────────────────────────────────────────
# Triton kernel
# ──────────────────────────────────────────────────────────────────────────
if _HAS_TRITON and _HAS_TORCH:

    @triton.jit
    def _grpo_forward_kernel(
        # Pointers
        policy_lp_ptr,
        old_lp_ptr,
        ref_lp_ptr,
        advantages_ptr,
        vision_mask_ptr,
        seq_lengths_ptr,
        out_loss_ptr,
        out_n_tokens_ptr,
        # Sizes
        L: tl.constexpr,
        # Hyper-params
        epsilon: tl.constexpr,
        beta: tl.constexpr,
        # Block size (tunable)
        BLOCK_L: tl.constexpr,
    ):
        """One kernel instance per sequence (row). Accumulates loss over text tokens."""
        row = tl.program_id(0)

        seq_len = tl.load(seq_lengths_ptr + row).to(tl.int32)
        advantage = tl.load(advantages_ptr + row)

        row_base = row * L

        acc_loss = tl.zeros([1], dtype=tl.float32)
        acc_n = tl.zeros([1], dtype=tl.int32)

        for block_start in range(0, seq_len, BLOCK_L):
            offsets = block_start + tl.arange(0, BLOCK_L)
            mask = offsets < seq_len

            # Load token-level data
            pol_lp = tl.load(policy_lp_ptr + row_base + offsets,
                              mask=mask, other=0.0).to(tl.float32)
            old_lp = tl.load(old_lp_ptr + row_base + offsets,
                              mask=mask, other=0.0).to(tl.float32)
            ref_lp = tl.load(ref_lp_ptr + row_base + offsets,
                              mask=mask, other=0.0).to(tl.float32)
            vmask  = tl.load(vision_mask_ptr + row_base + offsets,
                              mask=mask, other=0).to(tl.int32)

            # Combine: only count text tokens (vmask==1) and within seq_len
            active = mask & (vmask == 1)

            # ratio = exp(pol_lp - old_lp)
            log_ratio = pol_lp - old_lp
            ratio = tl.exp(log_ratio)

            # Clipped surrogate
            ratio_clipped = tl.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
            surr1 = ratio * advantage
            surr2 = ratio_clipped * advantage
            surrogate = -tl.minimum(surr1, surr2)

            # Approximated reverse KL:  exp(ref - pol) - (ref - pol) - 1
            diff = ref_lp - pol_lp
            kl = tl.exp(diff) - diff - 1.0

            token_loss = surrogate + beta * kl

            # Accumulate only over active (text) tokens
            acc_loss += tl.sum(tl.where(active, token_loss, tl.zeros_like(token_loss)), axis=0)
            acc_n    += tl.sum(tl.where(active, tl.ones_like(vmask), tl.zeros_like(vmask)), axis=0)

        tl.store(out_loss_ptr + row,    tl.sum(acc_loss, axis=0))
        tl.store(out_n_tokens_ptr + row, tl.sum(acc_n, axis=0).to(tl.float32))

    @triton.jit
    def _grpo_backward_kernel(
        # Forward inputs (needed for grad computation)
        policy_lp_ptr,
        old_lp_ptr,
        ref_lp_ptr,
        advantages_ptr,
        vision_mask_ptr,
        seq_lengths_ptr,
        n_tokens_ptr,
        grad_output_ptr,
        # Output: grad w.r.t. policy_log_probs
        grad_pol_ptr,
        L: tl.constexpr,
        epsilon: tl.constexpr,
        beta: tl.constexpr,
        BLOCK_L: tl.constexpr,
    ):
        """Backward kernel: ∂loss/∂policy_log_probs."""
        row = tl.program_id(0)

        seq_len = tl.load(seq_lengths_ptr + row).to(tl.int32)
        advantage = tl.load(advantages_ptr + row)
        n_tok = tl.load(n_tokens_ptr + row)
        grad_out = tl.load(grad_output_ptr + row)  # upstream grad for this seq

        row_base = row * L

        for block_start in range(0, seq_len, BLOCK_L):
            offsets = block_start + tl.arange(0, BLOCK_L)
            mask = offsets < seq_len

            pol_lp = tl.load(policy_lp_ptr + row_base + offsets,
                              mask=mask, other=0.0).to(tl.float32)
            old_lp = tl.load(old_lp_ptr + row_base + offsets,
                              mask=mask, other=0.0).to(tl.float32)
            ref_lp = tl.load(ref_lp_ptr + row_base + offsets,
                              mask=mask, other=0.0).to(tl.float32)
            vmask  = tl.load(vision_mask_ptr + row_base + offsets,
                              mask=mask, other=0).to(tl.int32)

            active = mask & (vmask == 1)

            log_ratio = pol_lp - old_lp
            ratio = tl.exp(log_ratio)
            ratio_clipped = tl.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

            # ∂surrogate/∂pol_lp
            # surrogate = -min(ratio*A, clipped*A)
            # When ratio is clipped: grad from surrogate = 0 w.r.t. pol_lp
            # When not clipped:      grad = -A * ratio  (chain rule: d(ratio)/d(pol_lp) = ratio)
            unclipped = (ratio > (1.0 - epsilon)) & (ratio < (1.0 + epsilon))
            # Actually: d(-min(r*A, rc*A))/d(pol_lp)
            # = d(-min(r*A, rc*A))/d(r) * dr/d(pol_lp)
            # dr/d(pol_lp) = ratio
            # d(-min(r*A, rc*A))/d(r):
            #   if r*A <= rc*A (not clipped and A>=0, or clipped and A<0): = -A
            #   else: = 0
            is_unclipped_active = (ratio * advantage <= ratio_clipped * advantage)
            d_surr_d_pol = tl.where(
                is_unclipped_active,
                -advantage * ratio,
                tl.zeros_like(ratio)
            )

            # ∂kl/∂pol_lp  where kl = exp(ref_lp - pol_lp) - (ref_lp - pol_lp) - 1
            # d(kl)/d(pol_lp) = -exp(ref_lp - pol_lp) + 1
            d_kl_d_pol = -tl.exp(ref_lp - pol_lp) + 1.0

            d_loss_d_pol = d_surr_d_pol + beta * d_kl_d_pol

            # Scale by upstream grad and divide by n_tokens
            scale = grad_out / (n_tok + 1e-8)
            grad = tl.where(active, d_loss_d_pol * scale, tl.zeros_like(d_loss_d_pol))

            tl.store(grad_pol_ptr + row_base + offsets, grad, mask=mask)

# ──────────────────────────────────────────────────────────────────────────
# PyTorch fallback (used when Triton is not available)
# ──────────────────────────────────────────────────────────────────────────
def _grpo_loss_pytorch(
    policy_log_probs: "torch.Tensor",
    old_log_probs: "torch.Tensor",
    ref_log_probs: "torch.Tensor",
    advantages: "torch.Tensor",
    vision_mask: "torch.Tensor",
    seq_lengths: "torch.Tensor",
    epsilon: float = 0.2,
    beta: float = 0.01,
) -> "torch.Tensor":
    """Pure-PyTorch GRPO loss.  Numerically identical to the Triton kernel.

    Returns a scalar loss tensor with gradient tracked through
    ``policy_log_probs`` only.
    """
    import torch as _torch

    BG, L = policy_log_probs.shape

    # Build validity mask: shape (BG, L)
    arange = _torch.arange(L, device=policy_log_probs.device).unsqueeze(0)  # (1, L)
    valid_mask = (arange < seq_lengths.unsqueeze(1))  # (BG, L)
    text_mask = valid_mask & vision_mask.bool()        # skip vision tokens

    # Ratio and surrogate
    log_ratio = policy_log_probs - old_log_probs
    ratio = _torch.exp(log_ratio)
    ratio_clipped = ratio.clamp(1.0 - epsilon, 1.0 + epsilon)

    adv = advantages.unsqueeze(1)  # (BG, 1)
    surr1 = ratio * adv
    surr2 = ratio_clipped * adv
    surrogate = -_torch.min(surr1, surr2)

    # Reverse KL approximation
    diff = ref_log_probs - policy_log_probs
    kl = _torch.exp(diff) - diff - 1.0

    token_loss = surrogate + beta * kl

    # Mask + mean
    token_loss = token_loss * text_mask.float()
    n_tokens = text_mask.float().sum(dim=1).clamp(min=1.0)  # (BG,)
    seq_loss = token_loss.sum(dim=1) / n_tokens             # (BG,)
    return seq_loss.mean()


# ──────────────────────────────────────────────────────────────────────────
# torch.autograd.Function wrapper
# ──────────────────────────────────────────────────────────────────────────
if _HAS_TORCH:

    class GRPOLossFunction(torch.autograd.Function):
        """Fused forward + backward for GRPO loss.

        Falls back to PyTorch autograd when Triton is not available.
        """

        @staticmethod
        def forward(
            ctx,
            policy_log_probs: "torch.Tensor",
            old_log_probs: "torch.Tensor",
            ref_log_probs: "torch.Tensor",
            advantages: "torch.Tensor",
            vision_mask: "torch.Tensor",
            seq_lengths: "torch.Tensor",
            epsilon: float,
            beta: float,
        ) -> "torch.Tensor":

            if _HAS_TRITON and policy_log_probs.is_cuda:
                BG, L = policy_log_probs.shape
                BLOCK_L = min(triton.next_power_of_2(L), 1024)

                out_loss    = torch.zeros(BG, device=policy_log_probs.device,
                                          dtype=torch.float32)
                out_n_tokens = torch.zeros(BG, device=policy_log_probs.device,
                                           dtype=torch.float32)

                # Cast inputs to float32 for kernel
                pol = policy_log_probs.float().contiguous()
                old = old_log_probs.float().contiguous()
                ref = ref_log_probs.float().contiguous()
                adv = advantages.float().contiguous()
                vm  = vision_mask.to(torch.int8).contiguous()
                sl  = seq_lengths.to(torch.int32).contiguous()

                grid = (BG,)
                _grpo_forward_kernel[grid](
                    pol, old, ref, adv, vm, sl,
                    out_loss, out_n_tokens,
                    L=L,
                    epsilon=epsilon,
                    beta=beta,
                    BLOCK_L=BLOCK_L,
                )

                # Scalar mean
                loss = (out_loss / out_n_tokens.clamp(min=1.0)).mean()

                ctx.save_for_backward(pol, old, ref, adv, vm, sl, out_n_tokens)
                ctx.epsilon = epsilon
                ctx.beta = beta
                ctx.L = L
                ctx.BLOCK_L = BLOCK_L
                ctx.use_triton = True
                return loss.to(policy_log_probs.dtype)

            else:
                # PyTorch fallback (supports non-CUDA or missing Triton)
                with torch.enable_grad():
                    pol_lp = policy_log_probs.detach().requires_grad_(True)
                    loss = _grpo_loss_pytorch(
                        pol_lp, old_log_probs, ref_log_probs,
                        advantages, vision_mask, seq_lengths, epsilon, beta
                    )
                ctx.save_for_backward(pol_lp, loss)
                ctx.use_triton = False
                return loss.detach()

        @staticmethod
        def backward(ctx, grad_output):
            if ctx.use_triton:
                pol, old, ref, adv, vm, sl, n_tokens = ctx.saved_tensors
                BG, L = pol.shape
                BLOCK_L = ctx.BLOCK_L

                grad_pol = torch.zeros_like(pol)

                # Per-sequence upstream grad = grad_output / BG  (mean backward)
                grad_per_seq = grad_output.expand(BG) / BG

                _grpo_backward_kernel[(BG,)](
                    pol, old, ref, adv, vm, sl,
                    n_tokens,
                    grad_per_seq.contiguous(),
                    grad_pol,
                    L=L,
                    epsilon=ctx.epsilon,
                    beta=ctx.beta,
                    BLOCK_L=BLOCK_L,
                )
                return (grad_pol.to(pol.dtype), None, None, None,
                        None, None, None, None)
            else:
                pol_lp, loss = ctx.saved_tensors
                # Recompute to get gradient (torch autograd fallback)
                # This is correct but not zero-copy; only used without Triton.
                grad = torch.autograd.grad(loss, pol_lp, grad_output)[0]
                return (grad, None, None, None, None, None, None, None)


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────
def grpo_loss(
    policy_log_probs: "torch.Tensor",
    old_log_probs: "torch.Tensor",
    ref_log_probs: "torch.Tensor",
    advantages: "torch.Tensor",
    vision_mask: "torch.Tensor",
    seq_lengths: "torch.Tensor",
    epsilon: float = 0.2,
    beta: float = 0.01,
) -> "torch.Tensor":
    """Compute the GRPO loss with optional vision-token masking.

    Parameters
    ----------
    policy_log_probs : Tensor, shape ``(B*G, L)``
        Token-level log-probabilities under the current policy for the chosen
        actions.  ``L`` is the padded sequence length.
    old_log_probs : Tensor, shape ``(B*G, L)``
        Log-probabilities under the old (frozen) policy snapshot.
    ref_log_probs : Tensor, shape ``(B*G, L)``
        Log-probabilities under the reference model (used for KL penalty).
    advantages : Tensor, shape ``(B*G,)``
        Per-sequence normalised advantages from group-relative reward
        normalisation.
    vision_mask : Tensor, shape ``(B*G, L)``, dtype uint8 / bool
        ``1`` for tokens that should contribute to the loss (response text).
        ``0`` for image patch tokens, system prompt, and question tokens.
    seq_lengths : Tensor, shape ``(B*G,)``, dtype int32 / int64
        Number of valid tokens per sequence (before padding).
    epsilon : float
        PPO clip ratio (default 0.2).
    beta : float
        KL penalty weight (default 0.01).

    Returns
    -------
    Tensor
        Scalar loss.
    """
    if not _HAS_TORCH:
        raise RuntimeError("grpo_loss requires PyTorch.")

    if _HAS_TRITON and policy_log_probs.is_cuda:
        return GRPOLossFunction.apply(
            policy_log_probs, old_log_probs, ref_log_probs,
            advantages, vision_mask, seq_lengths, epsilon, beta
        )
    else:
        return _grpo_loss_pytorch(
            policy_log_probs, old_log_probs, ref_log_probs,
            advantages, vision_mask, seq_lengths, epsilon, beta
        )
