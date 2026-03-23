"""
aerorl/kernels/cispo_loss.py
=============================
Fused Triton kernel for Clipped Importance-Sampled Policy Optimisation
(CISPO) loss.

CISPO replaces the token-level importance-ratio clip in GRPO with a
*sequence-level* clip on the product of per-token ratios.  This avoids
variance explosion in long sequences while still providing a tight trust
region.

Algorithm
---------
  seq_ratio  = exp( Σ_t  (pol_lp_t - old_lp_t) )   [product of per-token ratios]
               = exp( sum of log-ratios over text tokens )

  clipped_r  = clamp(seq_ratio, 1 - ε_seq, 1 + ε_seq)

  surrogate  = -min(seq_ratio * A,  clipped_r * A)   [per sequence]

  token_kl_t = exp(ref_lp_t - pol_lp_t) - (ref_lp_t - pol_lp_t) - 1

  loss       = mean_B*G [ surrogate + β * mean_text_tokens(token_kl) ]

Additional parameter vs GRPO
-----------------------------
epsilon_seq : float, default 0.2
    Clip applied to the *sequence-level* ratio.  Typically the same as the
    token-level epsilon used in GRPO but applied to the product ratio.
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
# Triton kernel
# ──────────────────────────────────────────────────────────────────────────
if _HAS_TRITON and _HAS_TORCH:

    @triton.jit
    def _cispo_forward_kernel(
        policy_lp_ptr,
        old_lp_ptr,
        ref_lp_ptr,
        advantages_ptr,
        vision_mask_ptr,
        seq_lengths_ptr,
        out_loss_ptr,
        out_n_tokens_ptr,
        L: tl.constexpr,
        epsilon_seq: tl.constexpr,
        beta: tl.constexpr,
        BLOCK_L: tl.constexpr,
    ):
        row = tl.program_id(0)

        seq_len   = tl.load(seq_lengths_ptr + row).to(tl.int32)
        advantage = tl.load(advantages_ptr + row)
        row_base  = row * L

        # Pass 1: accumulate sum of log-ratios over text tokens (for seq_ratio)
        sum_log_ratio = tl.zeros([1], dtype=tl.float32)
        acc_kl        = tl.zeros([1], dtype=tl.float32)
        acc_n         = tl.zeros([1], dtype=tl.int32)

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

            log_r = tl.where(active, pol_lp - old_lp, tl.zeros_like(pol_lp))
            sum_log_ratio += tl.sum(log_r, axis=0)

            diff = ref_lp - pol_lp
            kl   = tl.exp(diff) - diff - 1.0
            acc_kl += tl.sum(tl.where(active, kl, tl.zeros_like(kl)), axis=0)
            acc_n  += tl.sum(tl.where(active, tl.ones_like(vmask),
                                       tl.zeros_like(vmask)), axis=0)

        # Sequence-level clip
        seq_ratio = tl.exp(tl.sum(sum_log_ratio, axis=0))
        seq_ratio_clipped = tl.clamp(seq_ratio,
                                      1.0 - epsilon_seq,
                                      1.0 + epsilon_seq)

        surr1      = seq_ratio * advantage
        surr2      = seq_ratio_clipped * advantage
        surrogate  = -tl.minimum(surr1, surr2)

        mean_kl    = tl.sum(acc_kl, axis=0) / (tl.sum(acc_n, axis=0).to(tl.float32) + 1e-8)
        seq_loss   = surrogate + beta * mean_kl

        tl.store(out_loss_ptr + row,     seq_loss)
        tl.store(out_n_tokens_ptr + row, tl.sum(acc_n, axis=0).to(tl.float32))

    @triton.jit
    def _cispo_backward_kernel(
        policy_lp_ptr,
        old_lp_ptr,
        ref_lp_ptr,
        advantages_ptr,
        vision_mask_ptr,
        seq_lengths_ptr,
        # Precomputed forward quantities
        seq_ratios_ptr,        # (BG,) float32: exp(sum log-ratio)
        n_tokens_ptr,          # (BG,) float32
        grad_output_ptr,       # (BG,) upstream grad per seq
        # Output
        grad_pol_ptr,
        L: tl.constexpr,
        epsilon_seq: tl.constexpr,
        beta: tl.constexpr,
        BLOCK_L: tl.constexpr,
    ):
        row = tl.program_id(0)

        seq_len    = tl.load(seq_lengths_ptr + row).to(tl.int32)
        advantage  = tl.load(advantages_ptr + row)
        seq_ratio  = tl.load(seq_ratios_ptr + row)
        n_tok      = tl.load(n_tokens_ptr + row)
        grad_out   = tl.load(grad_output_ptr + row)
        row_base   = row * L

        # d(-min(r*A, rc*A))/d(r)
        is_unclipped = (seq_ratio > (1.0 - epsilon_seq)) & (seq_ratio < (1.0 + epsilon_seq))
        is_active_surr = (seq_ratio * advantage <= tl.clamp(seq_ratio, 1.0 - epsilon_seq, 1.0 + epsilon_seq) * advantage)
        d_surr_d_r = tl.where(is_active_surr, -advantage, 0.0)
        # d(r)/d(log_ratio_token) = r for each token (chain rule of product)
        # d(-min)/d(pol_lp_t) = d_surr_d_r * seq_ratio

        for block_start in range(0, seq_len, BLOCK_L):
            offsets = block_start + tl.arange(0, BLOCK_L)
            mask    = offsets < seq_len

            pol_lp = tl.load(policy_lp_ptr + row_base + offsets,
                              mask=mask, other=0.0).to(tl.float32)
            ref_lp = tl.load(ref_lp_ptr + row_base + offsets,
                              mask=mask, other=0.0).to(tl.float32)
            vmask  = tl.load(vision_mask_ptr + row_base + offsets,
                              mask=mask, other=0).to(tl.int32)
            active = mask & (vmask == 1)

            # Surrogate gradient w.r.t. pol_lp_t
            d_surr_d_pol = d_surr_d_r * seq_ratio  # chain: d(r)/d(sum_log_ratio) * d(sum_log_ratio)/d(pol_lp_t) = seq_ratio * 1

            # KL gradient
            d_kl_d_pol = -tl.exp(ref_lp - pol_lp) + 1.0

            d_loss_d_pol = d_surr_d_pol + beta * d_kl_d_pol / (n_tok + 1e-8)

            grad = tl.where(active, d_loss_d_pol * grad_out, tl.zeros_like(d_loss_d_pol))
            tl.store(grad_pol_ptr + row_base + offsets, grad, mask=mask)


# ──────────────────────────────────────────────────────────────────────────
# PyTorch fallback
# ──────────────────────────────────────────────────────────────────────────
def _cispo_loss_pytorch(
    policy_log_probs: "torch.Tensor",
    old_log_probs: "torch.Tensor",
    ref_log_probs: "torch.Tensor",
    advantages: "torch.Tensor",
    vision_mask: "torch.Tensor",
    seq_lengths: "torch.Tensor",
    epsilon_seq: float,
    beta: float,
) -> "torch.Tensor":
    import torch as _torch

    BG, L = policy_log_probs.shape
    arange     = _torch.arange(L, device=policy_log_probs.device).unsqueeze(0)
    valid_mask = arange < seq_lengths.unsqueeze(1)
    text_mask  = valid_mask & vision_mask.bool()

    # Sequence-level log-ratio (sum over text tokens)
    log_ratio     = (policy_log_probs - old_log_probs) * text_mask.float()
    sum_log_ratio = log_ratio.sum(dim=1)           # (BG,)
    seq_ratio     = _torch.exp(sum_log_ratio)

    seq_ratio_clipped = seq_ratio.clamp(1.0 - epsilon_seq, 1.0 + epsilon_seq)
    surr1 = seq_ratio * advantages
    surr2 = seq_ratio_clipped * advantages
    surrogate = -_torch.min(surr1, surr2)          # (BG,)

    # Token-level KL, averaged over text tokens
    diff = ref_log_probs - policy_log_probs
    kl   = _torch.exp(diff) - diff - 1.0
    kl   = kl * text_mask.float()
    n_tokens  = text_mask.float().sum(dim=1).clamp(min=1.0)
    mean_kl   = kl.sum(dim=1) / n_tokens           # (BG,)

    seq_loss = surrogate + beta * mean_kl
    return seq_loss.mean()


# ──────────────────────────────────────────────────────────────────────────
# torch.autograd.Function
# ──────────────────────────────────────────────────────────────────────────
if _HAS_TORCH:

    class CISPOLossFunction(torch.autograd.Function):

        @staticmethod
        def forward(ctx, policy_log_probs, old_log_probs, ref_log_probs,
                    advantages, vision_mask, seq_lengths, epsilon_seq, beta):

            if _HAS_TRITON and policy_log_probs.is_cuda:
                BG, L   = policy_log_probs.shape
                BLOCK_L = min(triton.next_power_of_2(L), 1024)

                pol = policy_log_probs.float().contiguous()
                old = old_log_probs.float().contiguous()
                ref = ref_log_probs.float().contiguous()
                adv = advantages.float().contiguous()
                vm  = vision_mask.to(torch.int8).contiguous()
                sl  = seq_lengths.to(torch.int32).contiguous()

                out_loss     = torch.zeros(BG, device=pol.device,
                                           dtype=torch.float32)
                out_n_tokens = torch.zeros(BG, device=pol.device,
                                           dtype=torch.float32)

                _cispo_forward_kernel[(BG,)](
                    pol, old, ref, adv, vm, sl,
                    out_loss, out_n_tokens,
                    L=L, epsilon_seq=epsilon_seq, beta=beta, BLOCK_L=BLOCK_L,
                )

                loss = out_loss.mean()

                # Store seq_ratios for backward (recompute inline for simplicity)
                arange    = torch.arange(L, device=pol.device).unsqueeze(0)
                valid     = arange < sl.unsqueeze(1)
                text_m    = valid & vm.bool()
                log_r     = (pol - old) * text_m.float()
                seq_ratios = torch.exp(log_r.sum(dim=1))

                ctx.save_for_backward(pol, old, ref, adv, vm, sl,
                                       seq_ratios, out_n_tokens)
                ctx.epsilon_seq = epsilon_seq
                ctx.beta        = beta
                ctx.L           = L
                ctx.BLOCK_L     = BLOCK_L
                ctx.use_triton  = True
                return loss.to(policy_log_probs.dtype)

            else:
                with torch.enable_grad():
                    pol_lp = policy_log_probs.detach().requires_grad_(True)
                    loss = _cispo_loss_pytorch(
                        pol_lp, old_log_probs, ref_log_probs,
                        advantages, vision_mask, seq_lengths,
                        epsilon_seq, beta,
                    )
                ctx.save_for_backward(pol_lp, loss)
                ctx.use_triton = False
                return loss.detach()

        @staticmethod
        def backward(ctx, grad_output):
            if ctx.use_triton:
                (pol, old, ref, adv, vm, sl,
                 seq_ratios, n_tokens) = ctx.saved_tensors
                BG = pol.shape[0]

                grad_pol     = torch.zeros_like(pol)
                # Per-seq upstream grad = grad_output / BG
                grad_per_seq = (grad_output / BG).expand(BG).contiguous()

                _cispo_backward_kernel[(BG,)](
                    pol, old, ref, adv, vm, sl,
                    seq_ratios, n_tokens, grad_per_seq, grad_pol,
                    L=ctx.L, epsilon_seq=ctx.epsilon_seq, beta=ctx.beta,
                    BLOCK_L=ctx.BLOCK_L,
                )
                return (grad_pol.to(pol.dtype),
                        None, None, None, None, None, None, None)
            else:
                pol_lp, loss = ctx.saved_tensors
                grad = torch.autograd.grad(loss, pol_lp, grad_output)[0]
                return (grad, None, None, None, None, None, None, None)


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────
def cispo_loss(
    policy_log_probs: "torch.Tensor",
    old_log_probs: "torch.Tensor",
    ref_log_probs: "torch.Tensor",
    advantages: "torch.Tensor",
    vision_mask: "torch.Tensor",
    seq_lengths: "torch.Tensor",
    epsilon_seq: float = 0.2,
    beta: float = 0.01,
) -> "torch.Tensor":
    """CISPO loss: sequence-level clipped importance-sampled PO.

    Parameters
    ----------
    policy_log_probs, old_log_probs, ref_log_probs, advantages,
    vision_mask, seq_lengths, beta
        Same semantics as :func:`aerorl.kernels.grpo_loss.grpo_loss`.
    epsilon_seq : float
        Clip bound applied to the *product* of per-token importance ratios
        (i.e. the sequence-level ratio).
    """
    if not _HAS_TORCH:
        raise RuntimeError("cispo_loss requires PyTorch.")

    if _HAS_TRITON and policy_log_probs.is_cuda:
        return CISPOLossFunction.apply(
            policy_log_probs, old_log_probs, ref_log_probs,
            advantages, vision_mask, seq_lengths, epsilon_seq, beta,
        )
    else:
        return _cispo_loss_pytorch(
            policy_log_probs, old_log_probs, ref_log_probs,
            advantages, vision_mask, seq_lengths, epsilon_seq, beta,
        )
