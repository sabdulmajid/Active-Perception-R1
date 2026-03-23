"""
aerorl — Zero-copy VLM RL library.

Drop-in library that eliminates KV cache + logits duplication between vLLM
rollout and PyTorch training for Vision-Language Models.  Enables 2× context
or 2× group size on a single 96 GB GPU for GRPO/GSPO/CISPO.

Quick start
-----------
::

    from aerorl import AeroRLConfig, wrap_vlm_for_rl

    config = AeroRLConfig(
        zero_copy_kv=True,
        mask_vision_tokens=True,
        quant_ref_bits=8,
    )

    policy, ref = wrap_vlm_for_rl("Qwen/Qwen2.5-VL-7B-Instruct", config)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__  = "AeroRL contributors"

from .config import AeroRLConfig
from .extensions.ipc_kv_cache import AeroRLSharedKVCache
from .kernels import grpo_loss, gspo_loss, cispo_loss
from .utils.vision_mask import (
    build_vision_mask_from_labels,
    build_vision_mask_from_input_ids,
    build_vision_mask_auto,
    compact_vision_mask,
)
from .utils.quant_ref import QuantisedRefModel, BackgroundRefHook
from .utils.processor_utils import VisionMaskBuilder, detect_family
from .kernels.dapo_filter import apply_dapo_filter, dapo_variance_mask


def wrap_vlm_for_rl(
    model_name_or_path: str,
    config: "AeroRLConfig",
    policy_device_map: str = "auto",
    ref_device_map: str = "auto",
    **hf_kwargs,
):
    """Load a VLM policy model and an optional quantised reference model.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model ID or local path.
    config : AeroRLConfig
        AeroRL configuration.
    policy_device_map : str
        ``device_map`` arg for the policy model (default ``"auto"``).
    ref_device_map : str
        ``device_map`` arg for the reference model (default ``"auto"``).
    **hf_kwargs:
        Extra kwargs forwarded to both ``from_pretrained`` calls (e.g.
        ``trust_remote_code=True``).

    Returns
    -------
    Tuple[policy_model, ref_model | None]
        ``policy_model`` is a HuggingFace ``PreTrainedModel`` ready for training.
        ``ref_model`` is a :class:`QuantisedRefModel` when
        ``config.quant_ref_bits > 0``, otherwise ``None``.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "wrap_vlm_for_rl requires `torch` and `transformers`.  "
            "Install with: pip install torch transformers"
        ) from exc

    _dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    torch_dtype = _dtype_map.get(config.dtype, torch.bfloat16)

    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=policy_device_map,
        **hf_kwargs,
    )
    policy_model.train()

    ref_model = None
    if config.use_quant_ref:
        ref_model = QuantisedRefModel.from_pretrained(
            model_name_or_path,
            quant_bits=config.quant_ref_bits,
            backend=config.quant_ref_backend,
            device_map=ref_device_map,
            torch_dtype=torch_dtype,
            **hf_kwargs,
        )

    return policy_model, ref_model


__all__ = [
    # Core API
    "AeroRLConfig",
    "wrap_vlm_for_rl",
    # KV cache
    "AeroRLSharedKVCache",
    # Loss kernels
    "grpo_loss",
    "gspo_loss",
    "cispo_loss",
    # Vision masking
    "build_vision_mask_from_labels",
    "build_vision_mask_from_input_ids",
    "build_vision_mask_auto",
    "compact_vision_mask",
    "VisionMaskBuilder",
    "detect_family",
    # Reference model
    "QuantisedRefModel",
    "BackgroundRefHook",
    # DAPO filter
    "apply_dapo_filter",
    "dapo_variance_mask",
]
