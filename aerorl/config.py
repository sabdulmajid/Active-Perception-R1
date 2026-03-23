"""AeroRL configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class AeroRLConfig:
    """Top-level configuration for the AeroRL library.

    Parameters
    ----------
    zero_copy_kv:
        Enable CUDA IPC zero-copy KV-cache sharing between the vLLM rollout
        process and the PyTorch training process.  Requires the compiled CUDA
        extension (``aerorl_ipc_ext``).  Falls back to a reference-copy path
        when the extension is not available.

    mask_vision_tokens:
        Skip vision / prompt tokens when computing policy-gradient losses so
        that only generated text tokens contribute to the gradient.

    loss_type:
        Which policy-gradient algorithm to use.  One of ``"grpo"``, ``"gspo"``,
        or ``"cispo"``.

    epsilon:
        PPO-style clip ratio.  Ratio is clamped to ``[1 - epsilon, 1 + epsilon]``.

    beta:
        Weight of the KL-divergence penalty (applied only on text tokens when
        ``mask_vision_tokens=True``).

    quant_ref_bits:
        Bit-width for the quantised reference model (``8`` for INT8, ``0`` to
        disable quantisation).  Values of ``4`` are also accepted when a
        bitsandbytes backend with 4-bit support is present.

    quant_ref_backend:
        Quantisation backend to use.  ``"bitsandbytes"`` (default) or
        ``"torchao"``.

    dapo_variance_filter:
        Enable the DAPO variance filter: sequences whose within-group reward
        variance falls below ``dapo_min_variance`` are skipped.

    dapo_min_variance:
        Minimum reward variance threshold for the DAPO filter.  Groups with
        variance strictly below this value are excluded from the loss.

    vision_processor:
        Override the VLM processor type for vision-mask auto-detection.
        Supported values: ``"auto"``, ``"qwen2_5_vl"``, ``"llava_1_6"``,
        ``"internvl2"``, ``"phi3_vision"``.  ``"auto"`` infers from the model
        class name at runtime.

    max_seq_len:
        Maximum sequence length (tokens) used for kernel block-size hints.

    dtype:
        Data type string passed to the Triton kernels (``"fp16"``, ``"bf16"``,
        or ``"fp32"``).
    """

    # ── Zero-copy KV sharing ────────────────────────────────────────────────
    zero_copy_kv: bool = True

    # ── Vision masking ──────────────────────────────────────────────────────
    mask_vision_tokens: bool = True

    # ── Loss algorithm ──────────────────────────────────────────────────────
    loss_type: Literal["grpo", "gspo", "cispo"] = "grpo"

    # ── PPO-style clipping ──────────────────────────────────────────────────
    epsilon: float = 0.2

    # ── KL penalty ─────────────────────────────────────────────────────────
    beta: float = 0.01

    # ── Quantised reference model ───────────────────────────────────────────
    quant_ref_bits: int = 8
    quant_ref_backend: Literal["bitsandbytes", "torchao", "none"] = "bitsandbytes"

    # ── DAPO variance filter (Phase 2) ──────────────────────────────────────
    dapo_variance_filter: bool = False
    dapo_min_variance: float = 1e-4

    # ── Vision processor ────────────────────────────────────────────────────
    vision_processor: Literal[
        "auto", "qwen2_5_vl", "llava_1_6", "internvl2", "phi3_vision"
    ] = "auto"

    # ── Misc ────────────────────────────────────────────────────────────────
    max_seq_len: int = 8192
    dtype: Literal["fp16", "bf16", "fp32"] = "bf16"

    # ── Internal / advanced ─────────────────────────────────────────────────
    _extra: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.quant_ref_bits not in (0, 4, 8):
            raise ValueError(
                f"quant_ref_bits must be 0, 4, or 8; got {self.quant_ref_bits}"
            )
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError(
                f"epsilon must be in (0, 1); got {self.epsilon}"
            )
        if self.beta < 0.0:
            raise ValueError(f"beta must be >= 0; got {self.beta}")

    # ── Convenience helpers ─────────────────────────────────────────────────
    @property
    def use_quant_ref(self) -> bool:
        return self.quant_ref_bits > 0

    @property
    def triton_dtype(self) -> str:
        return self.dtype

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)
