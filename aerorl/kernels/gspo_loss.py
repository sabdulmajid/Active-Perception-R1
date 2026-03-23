"""
aerorl/kernels/gspo_loss.py
============================
Fused Triton kernel for Group Sparse Policy Optimisation (GSPO) loss.

GSPO extends GRPO with group-sparse advantage estimation: only the top-k
sequences within each group (by absolute advantage magnitude) contribute to
the policy gradient.  The remaining sequences are detached, reducing gradient
noise and variance for large group sizes.

Additional parameter vs GRPO
-----------------------------
top_k_ratio : float, default 0.5
    Fraction of sequences within each *group of G* to keep.  E.g., with
    ``G=16`` and ``top_k_ratio=0.5``, only the 8 sequences with the largest
    |advantage| are used.  Set to 1.0 to recover standard GRPO.

All other parameters are identical to :mod:`aerorl.kernels.grpo_loss`.
"""

from __future__ import annotations

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
# Triton kernel  (identical to GRPO kernel; sparse selection is done on host)
# ──────────────────────────────────────────────────────────────────────────
if _HAS_TRITON and _HAS_TORCH:

    @triton.jit
    def _gspo_forward_kernel(
        policy_lp_ptr,
        old_lp_ptr,
        ref_lp_ptr,
        advantages_ptr,
        active_seq_ptr,        # (B*G,) uint8: 1 = this seq participates
        vision_mask_ptr,
        seq_lengths_ptr,
        out_loss_ptr,
        out_n_tokens_ptr,
        L: tl.constexpr,
        epsilon: tl.constexpr,
        beta: tl.constexpr,
        BLOCK_L: tl.constexpr,
    ):
        row = tl.program_id(0)

        # Skip rows not selected by sparse filter
        is_active = tl.load(active_seq_ptr + row).to(tl.int32)
        if is_active == 0:
            tl.store(out_loss_ptr + row, 0.0)
            tl.store(out_n_tokens_ptr + row, 0.0)
            return

        seq_len   = tl.load(seq_lengths_ptr + row).to(tl.int32)
        advantage = tl.load(advantages_ptr + row)
        row_base  = row * L

        acc_loss = tl.zeros([1], dtype=tl.float32)
        acc_n    = tl.zeros([1], dtype=tl.int32)

        for block_start in range(0, seq_len, BLOCK_L):
            offsets = block_start + tl.arange(0, BLOCK_L)
            mask    = offsets < seq_len

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
            ratio         = tl.exp(log_ratio)
            ratio_clipped = tl.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
            surr1 = ratio * advantage
            surr2 = ratio_clipped * advantage
            surrogate = -tl.minimum(surr1, surr2)

            diff = ref_lp - pol_lp
            kl   = tl.exp(diff) - diff - 1.0

            token_loss = surrogate + beta * kl

            acc_loss += tl.sum(tl.where(active, token_loss,
                                         tl.zeros_like(token_loss)), axis=0)
            acc_n    += tl.sum(tl.where(active, tl.ones_like(vmask),
                                         tl.zeros_like(vmask)), axis=0)

        tl.store(out_loss_ptr + row,     tl.sum(acc_loss, axis=0))
        tl.store(out_n_tokens_ptr + row, tl.sum(acc_n, axis=0).to(tl.float32))


# ──────────────────────────────────────────────────────────────────────────
# Host-side sparse selection helper
# ──────────────────────────────────────────────────────────────────────────
def _build_sparse_mask(advantages: "torch.Tensor", G: int, top_k_ratio: float):
    """Return a boolean mask (B*G,) selecting top-k sequences per group.

    Parameters
    ----------
    advantages : Tensor, shape ``(B*G,)``
    G : int
        Group size (number of sequences per prompt).
    top_k_ratio : float
        Fraction to keep.  Clamped to ``[1/G, 1.0]``.
    """
    import torch as _torch

    BG = advantages.shape[0]
    B  = BG // G
    k  = max(1, round(G * top_k_ratio))

    # Reshape to (B, G), select top-k per row by |advantage|
    adv_grouped = advantages.view(B, G)
    abs_adv     = adv_grouped.abs()
    _, top_idx  = abs_adv.topk(k, dim=1, largest=True, sorted=False)  # (B, k)

    sparse_mask = _torch.zeros(B, G, dtype=_torch.uint8,
                               device=advantages.device)
    sparse_mask.scatter_(1, top_idx, 1)
    return sparse_mask.view(BG)


# ──────────────────────────────────────────────────────────────────────────
# PyTorch fallback
# ──────────────────────────────────────────────────────────────────────────
def _gspo_loss_pytorch(
    policy_log_probs: "torch.Tensor",
    old_log_probs: "torch.Tensor",
    ref_log_probs: "torch.Tensor",
    advantages: "torch.Tensor",
    vision_mask: "torch.Tensor",
    seq_lengths: "torch.Tensor",
    sparse_mask: "torch.Tensor",
    epsilon: float,
    beta: float,
) -> "torch.Tensor":
    import torch as _torch

    BG, L = policy_log_probs.shape

    arange     = _torch.arange(L, device=policy_log_probs.device).unsqueeze(0)
    valid_mask = arange < seq_lengths.unsqueeze(1)
    text_mask  = valid_mask & vision_mask.bool()
    # Zero out inactive sequences
    text_mask  = text_mask & sparse_mask.bool().unsqueeze(1)

    log_ratio     = policy_log_probs - old_log_probs
    ratio         = _torch.exp(log_ratio)
    ratio_clipped = ratio.clamp(1.0 - epsilon, 1.0 + epsilon)

    adv   = advantages.unsqueeze(1)
    surr1 = ratio * adv
    surr2 = ratio_clipped * adv
    surrogate = -_torch.min(surr1, surr2)

    diff = ref_log_probs - policy_log_probs
    kl   = _torch.exp(diff) - diff - 1.0

    token_loss = surrogate + beta * kl
    token_loss = token_loss * text_mask.float()

    # Normalise over active (text) tokens, average over active sequences
    n_tokens = text_mask.float().sum(dim=1).clamp(min=1.0)
    seq_loss = token_loss.sum(dim=1) / n_tokens

    # Weight only active sequences
    active_seqs = sparse_mask.float()
    n_active    = active_seqs.sum().clamp(min=1.0)
    return (seq_loss * active_seqs).sum() / n_active


# ──────────────────────────────────────────────────────────────────────────
# torch.autograd.Function
# ──────────────────────────────────────────────────────────────────────────
if _HAS_TORCH:

    class GSPOLossFunction(torch.autograd.Function):

        @staticmethod
        def forward(ctx, policy_log_probs, old_log_probs, ref_log_probs,
                    advantages, vision_mask, seq_lengths, sparse_mask,
                    epsilon, beta):

            if _HAS_TRITON and policy_log_probs.is_cuda:
                BG, L   = policy_log_probs.shape
                BLOCK_L = min(triton.next_power_of_2(L), 1024)

                pol = policy_log_probs.float().contiguous()
                old = old_log_probs.float().contiguous()
                ref = ref_log_probs.float().contiguous()
                adv = advantages.float().contiguous()
                vm  = vision_mask.to(torch.int8).contiguous()
                sl  = seq_lengths.to(torch.int32).contiguous()
                sm  = sparse_mask.to(torch.uint8).contiguous()

                out_loss     = torch.zeros(BG, device=pol.device,
                                           dtype=torch.float32)
                out_n_tokens = torch.zeros(BG, device=pol.device,
                                           dtype=torch.float32)

                _gspo_forward_kernel[(BG,)](
                    pol, old, ref, adv, sm, vm, sl,
                    out_loss, out_n_tokens,
                    L=L, epsilon=epsilon, beta=beta, BLOCK_L=BLOCK_L,
                )

                active_float = sm.float()
                n_active     = active_float.sum().clamp(min=1.0)
                loss = ((out_loss / out_n_tokens.clamp(min=1.0)) * active_float
                        ).sum() / n_active

                ctx.save_for_backward(pol, old, ref, adv, vm, sl, sm, out_n_tokens)
                ctx.epsilon = epsilon
                ctx.beta    = beta
                ctx.L       = L
                ctx.BLOCK_L = BLOCK_L
                ctx.use_triton = True
                return loss.to(policy_log_probs.dtype)

            else:
                with torch.enable_grad():
                    pol_lp = policy_log_probs.detach().requires_grad_(True)
                    loss = _gspo_loss_pytorch(
                        pol_lp, old_log_probs, ref_log_probs,
                        advantages, vision_mask, seq_lengths, sparse_mask,
                        epsilon, beta,
                    )
                ctx.save_for_backward(pol_lp, loss)
                ctx.use_triton = False
                return loss.detach()

        @staticmethod
        def backward(ctx, grad_output):
            if ctx.use_triton:
                pol, old, ref, adv, vm, sl, sm, n_tokens = ctx.saved_tensors
                BG = pol.shape[0]

                active_float = sm.float()
                n_active     = active_float.sum().clamp(min=1.0)

                # Reuse GRPO backward kernel (gradient is identical per active seq)
                from .grpo_loss import _grpo_backward_kernel

                grad_pol      = torch.zeros_like(pol)
                grad_per_seq  = (grad_output * active_float / n_active
                                 / (n_tokens + 1e-8)).contiguous()

                _grpo_backward_kernel[(BG,)](
                    pol, old, ref, adv, vm, sl,
                    n_tokens, grad_per_seq, grad_pol,
                    L=ctx.L, epsilon=ctx.epsilon, beta=ctx.beta,
                    BLOCK_L=ctx.BLOCK_L,
                )
                return (grad_pol.to(pol.dtype),
                        None, None, None, None, None, None, None, None)
            else:
                pol_lp, loss = ctx.saved_tensors
                grad = torch.autograd.grad(loss, pol_lp, grad_output)[0]
                return (grad, None, None, None, None, None, None, None, None)


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────
def gspo_loss(
    policy_log_probs: "torch.Tensor",
    old_log_probs: "torch.Tensor",
    ref_log_probs: "torch.Tensor",
    advantages: "torch.Tensor",
    vision_mask: "torch.Tensor",
    seq_lengths: "torch.Tensor",
    G: int = 8,
    top_k_ratio: float = 0.5,
    epsilon: float = 0.2,
    beta: float = 0.01,
) -> "torch.Tensor":
    """GSPO loss: GRPO with group-sparse advantage selection.

    Parameters
    ----------
    policy_log_probs, old_log_probs, ref_log_probs, advantages,
    vision_mask, seq_lengths, epsilon, beta
        Same semantics as :func:`aerorl.kernels.grpo_loss.grpo_loss`.
    G : int
        Number of sequences per prompt (group size).
    top_k_ratio : float
        Fraction of sequences per group to keep in the gradient.
        E.g. ``0.5`` with ``G=16`` keeps the 8 sequences with the largest
        ``|advantage|``.
    """
    if not _HAS_TORCH:
        raise RuntimeError("gspo_loss requires PyTorch.")

    sparse_mask = _build_sparse_mask(advantages, G=G, top_k_ratio=top_k_ratio)

    if _HAS_TRITON and policy_log_probs.is_cuda:
        return GSPOLossFunction.apply(
            policy_log_probs, old_log_probs, ref_log_probs,
            advantages, vision_mask, seq_lengths, sparse_mask,
            epsilon, beta,
        )
    else:
        return _gspo_loss_pytorch(
            policy_log_probs, old_log_probs, ref_log_probs,
            advantages, vision_mask, seq_lengths, sparse_mask,
            epsilon, beta,
        )
