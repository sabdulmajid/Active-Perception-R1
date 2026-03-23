"""
aerorl/kernels/dapo_filter.py
==============================
DAPO (Dynamic Advantage Policy Optimization) variance filter.

DAPO removes groups whose within-group reward variance falls below a
threshold before the policy-gradient update.  This filters out "easy" groups
where all sampled responses receive similar rewards — those groups contribute
mostly noise to the gradient.

This is integrated transparently into the loss functions when
``AeroRLConfig.dapo_variance_filter=True``.

Reference
---------
DAPO: An Open-Source LLM Reinforcement Learning System at Scale
https://arxiv.org/abs/2503.14476
"""

from __future__ import annotations

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


def dapo_variance_mask(
    rewards: "torch.Tensor",
    G: int,
    min_variance: float = 1e-4,
) -> "torch.Tensor":
    """Return a boolean group-level mask filtering low-variance groups.

    Parameters
    ----------
    rewards : Tensor, shape ``(B*G,)``
        Per-sequence scalar rewards (before advantage normalisation).
    G : int
        Group size (number of sequences per prompt).
    min_variance : float
        Groups whose reward variance is strictly below this threshold are
        masked out (mask = False).

    Returns
    -------
    Tensor, dtype bool, shape ``(B*G,)``
        True for sequences in groups that pass the variance filter.
        False for sequences in low-variance (uninformative) groups.
    """
    if not _HAS_TORCH:
        raise RuntimeError("dapo_variance_mask requires PyTorch.")
    import torch as _torch

    BG = rewards.shape[0]
    B  = BG // G

    rewards_grouped = rewards.view(B, G).float()           # (B, G)
    var             = rewards_grouped.var(dim=1, unbiased=False)  # (B,)

    group_keep = var >= min_variance                        # (B,)
    seq_keep   = group_keep.unsqueeze(1).expand(B, G).reshape(BG)  # (B*G,)
    return seq_keep


def apply_dapo_filter(
    rewards: "torch.Tensor",
    advantages: "torch.Tensor",
    G: int,
    min_variance: float = 1e-4,
) -> tuple:
    """Return ``(filtered_rewards, filtered_advantages, keep_mask)``.

    Sequences belonging to low-variance groups are zero-masked in both
    ``rewards`` and ``advantages``.  The caller can weight the loss by
    ``keep_mask.float()`` or simply pass the zeroed advantages directly.

    Parameters
    ----------
    rewards : Tensor, shape ``(B*G,)``
    advantages : Tensor, shape ``(B*G,)``
    G : int
    min_variance : float

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor]
        ``(rewards_out, advantages_out, keep_mask)`` — all shape ``(B*G,)``.
    """
    if not _HAS_TORCH:
        raise RuntimeError("apply_dapo_filter requires PyTorch.")
    import torch as _torch

    keep = dapo_variance_mask(rewards, G, min_variance)
    keep_float = keep.float()
    return rewards * keep_float, advantages * keep_float, keep
