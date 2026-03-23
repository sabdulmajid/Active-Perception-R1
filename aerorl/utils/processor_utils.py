"""
aerorl/utils/processor_utils.py
================================
Auto-detection and vision-mask helpers for supported VLM processor families.

Supported models (Phase 2)
--------------------------
- Qwen2.5-VL (``Qwen/Qwen2.5-VL-*``)
- LLaVA-1.6 (``llava-hf/llava-v1.6-*``)
- InternVL2 (``OpenGVLab/InternVL2-*``)
- Phi-3-Vision (``microsoft/Phi-3-vision-*``)

Each family requires slightly different logic to locate image-patch token
ranges inside the flat ``input_ids`` sequence.  This module encapsulates
those differences behind a unified ``VisionMaskBuilder`` class.

Usage
-----
::

    from aerorl.utils.processor_utils import VisionMaskBuilder

    builder = VisionMaskBuilder.from_model(model_name_or_path)
    vision_mask = builder.build(processor_output, response_start_ids)
"""

from __future__ import annotations

import re
from typing import Optional

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


# ──────────────────────────────────────────────────────────────────────────
# Family detection
# ──────────────────────────────────────────────────────────────────────────
_FAMILY_PATTERNS: list[tuple[str, str]] = [
    # (regex pattern for model name/path, canonical family name)
    (r"(?i)qwen2[._-]?5[._-]?vl",   "qwen2_5_vl"),
    (r"(?i)qwen2[._-]?vl",           "qwen2_5_vl"),
    (r"(?i)llava[._-]?v?1[._-]?6",  "llava_1_6"),
    (r"(?i)llava",                    "llava_1_6"),
    (r"(?i)internvl2",                "internvl2"),
    (r"(?i)internvl",                 "internvl2"),
    (r"(?i)phi[._-]?3[._-]?vision",  "phi3_vision"),
    (r"(?i)phi[._-]?3",               "phi3_vision"),
]


def detect_family(model_name_or_path: str) -> str:
    """Return the canonical VLM family name for *model_name_or_path*.

    Returns
    -------
    str
        One of ``"qwen2_5_vl"``, ``"llava_1_6"``, ``"internvl2"``,
        ``"phi3_vision"``, or ``"unknown"``.
    """
    for pattern, family in _FAMILY_PATTERNS:
        if re.search(pattern, model_name_or_path):
            return family
    return "unknown"


# ──────────────────────────────────────────────────────────────────────────
# Per-family image token IDs and segment-level helpers
# ──────────────────────────────────────────────────────────────────────────
class _Qwen25VLMaskBuilder:
    """Vision mask builder for Qwen2.5-VL.

    Qwen2.5-VL wraps image patches between ``<|vision_start|>`` (id 151652)
    and ``<|vision_end|>`` (id 151653).  All tokens within that span
    (inclusive of the markers) are image tokens.
    """

    IMAGE_PAD_ID    = 151655
    VISION_START_ID = 151652
    VISION_END_ID   = 151653

    def build(
        self,
        input_ids: "torch.Tensor",
        response_start_ids: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        import torch as _torch

        BG, L = input_ids.shape
        mask = _torch.ones(BG, L, dtype=_torch.uint8, device=input_ids.device)

        # Mark image pad tokens
        mask[input_ids == self.IMAGE_PAD_ID]    = 0
        mask[input_ids == self.VISION_START_ID] = 0
        mask[input_ids == self.VISION_END_ID]   = 0

        # Mask prompt region
        if response_start_ids is not None:
            positions = _torch.arange(L, device=input_ids.device).unsqueeze(0)
            mask[positions < response_start_ids.unsqueeze(1)] = 0

        return mask


class _LLaVA16MaskBuilder:
    """Vision mask builder for LLaVA-1.6.

    LLaVA-1.6 (LLaMA-based) uses token ID 32000 (``<image>``) as a single
    placeholder that is later *expanded* to multiple patch tokens by the
    visual tower.  In the flat input_ids, every occurrence of 32000 is an
    image region marker; we also mask its immediate neighbours if
    ``expand_neighbours=True`` (handles some tokeniser variants).
    """

    IMAGE_TOKEN_ID = 32000

    def build(
        self,
        input_ids: "torch.Tensor",
        response_start_ids: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        import torch as _torch

        BG, L = input_ids.shape
        mask = _torch.ones(BG, L, dtype=_torch.uint8, device=input_ids.device)
        mask[input_ids == self.IMAGE_TOKEN_ID] = 0

        if response_start_ids is not None:
            positions = _torch.arange(L, device=input_ids.device).unsqueeze(0)
            mask[positions < response_start_ids.unsqueeze(1)] = 0

        return mask


class _InternVL2MaskBuilder:
    """Vision mask builder for InternVL2.

    InternVL2 uses ``token_type_ids`` (0=text, 1=image) when available.
    Falls back to masking token ID 92543 (``<img>``).
    """

    IMAGE_TOKEN_ID = 92543

    def build(
        self,
        input_ids: "torch.Tensor",
        response_start_ids: Optional["torch.Tensor"] = None,
        token_type_ids: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        import torch as _torch

        BG, L = input_ids.shape
        if token_type_ids is not None:
            if not isinstance(token_type_ids, _torch.Tensor):
                token_type_ids = _torch.tensor(token_type_ids,
                                               device=input_ids.device)
            mask = (token_type_ids != 1).to(_torch.uint8)
        else:
            mask = _torch.ones(BG, L, dtype=_torch.uint8,
                               device=input_ids.device)
            mask[input_ids == self.IMAGE_TOKEN_ID] = 0

        if response_start_ids is not None:
            positions = _torch.arange(L, device=input_ids.device).unsqueeze(0)
            mask[positions < response_start_ids.unsqueeze(1)] = 0

        return mask


class _Phi3VisionMaskBuilder:
    """Vision mask builder for Phi-3-Vision.

    Phi-3-Vision uses ``<|image_1|>`` (id 32044) and subsequent patch tokens
    32045–32047 as image placeholders.
    """

    IMAGE_TOKEN_IDS = [32044, 32045, 32046, 32047]

    def build(
        self,
        input_ids: "torch.Tensor",
        response_start_ids: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        import torch as _torch

        BG, L = input_ids.shape
        mask = _torch.ones(BG, L, dtype=_torch.uint8, device=input_ids.device)
        for tid in self.IMAGE_TOKEN_IDS:
            mask[input_ids == tid] = 0

        if response_start_ids is not None:
            positions = _torch.arange(L, device=input_ids.device).unsqueeze(0)
            mask[positions < response_start_ids.unsqueeze(1)] = 0

        return mask


# ──────────────────────────────────────────────────────────────────────────
# Unified builder
# ──────────────────────────────────────────────────────────────────────────
class VisionMaskBuilder:
    """Unified vision-token mask builder for supported VLM families.

    Parameters
    ----------
    family : str
        VLM family: ``"qwen2_5_vl"``, ``"llava_1_6"``, ``"internvl2"``,
        ``"phi3_vision"``, or ``"unknown"``.
    """

    _BUILDERS = {
        "qwen2_5_vl":  _Qwen25VLMaskBuilder,
        "llava_1_6":   _LLaVA16MaskBuilder,
        "internvl2":   _InternVL2MaskBuilder,
        "phi3_vision": _Phi3VisionMaskBuilder,
    }

    def __init__(self, family: str = "unknown"):
        self.family = family
        builder_cls = self._BUILDERS.get(family)
        self._builder = builder_cls() if builder_cls is not None else None

    @classmethod
    def from_model(cls, model_name_or_path: str) -> "VisionMaskBuilder":
        """Instantiate a ``VisionMaskBuilder`` by auto-detecting the family.

        Parameters
        ----------
        model_name_or_path : str
            HuggingFace model ID or local directory path.
        """
        family = detect_family(model_name_or_path)
        return cls(family=family)

    def build(
        self,
        processor_output: dict,
        response_start_ids: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Build and return the vision mask.

        Parameters
        ----------
        processor_output : dict
            Must contain ``"input_ids"`` key.  May also contain
            ``"token_type_ids"`` (InternVL2), ``"labels"`` (verl convention).
        response_start_ids : Tensor, shape ``(B*G,)``, optional
            Token index at which the model response begins.  Tokens before
            this index are masked regardless of token type.

        Returns
        -------
        Tensor, dtype uint8, shape ``(B*G, L)``
        """
        if not _HAS_TORCH:
            raise RuntimeError("VisionMaskBuilder.build requires PyTorch.")
        import torch as _torch

        # Fast path: verl-style labels
        if "labels" in processor_output:
            labels = processor_output["labels"]
            if not isinstance(labels, _torch.Tensor):
                labels = _torch.tensor(labels)
            return (labels != -100).to(_torch.uint8)

        input_ids = processor_output["input_ids"]
        if not isinstance(input_ids, _torch.Tensor):
            input_ids = _torch.tensor(input_ids)

        if self._builder is None:
            # No family-specific builder; use generic label-like approach
            import warnings
            warnings.warn(
                f"VisionMaskBuilder: unknown family '{self.family}'.  "
                "All tokens will be treated as text tokens.",
                stacklevel=2,
            )
            BG, L = input_ids.shape
            return _torch.ones(BG, L, dtype=_torch.uint8,
                               device=input_ids.device)

        # InternVL2 may pass token_type_ids
        extra_kwargs = {}
        if self.family == "internvl2" and "token_type_ids" in processor_output:
            extra_kwargs["token_type_ids"] = processor_output["token_type_ids"]

        return self._builder.build(
            input_ids,
            response_start_ids=response_start_ids,
            **extra_kwargs,
        )

    def __repr__(self) -> str:
        return f"VisionMaskBuilder(family={self.family!r})"
