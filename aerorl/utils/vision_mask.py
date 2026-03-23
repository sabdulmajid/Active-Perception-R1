"""
aerorl/utils/vision_mask.py
============================
Compact vision-token mask helpers.

A *vision mask* is a boolean/uint8 tensor of shape ``(B*G, L)`` where:
  - ``1`` marks **text / response** tokens that contribute to the loss.
  - ``0`` marks image patch tokens, system-prompt tokens, and question tokens
    that should be skipped during policy-gradient loss reduction.

These helpers work in two modes:

1. **Processor-based** (preferred): extract token-type information directly
   from the tokeniser / processor output (``input_ids`` + ``image_token_id``
   or processor-specific segment IDs).

2. **Label-based** (verl-style): a ``labels`` tensor already has ``-100`` at
   positions that should be masked; the mask is trivially ``labels != -100``.
"""

from __future__ import annotations

from typing import Optional, Union

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


# ──────────────────────────────────────────────────────────────────────────
# Image-token IDs for supported models (fallback constants when the
# processor is not directly inspectable at mask-build time).
# ──────────────────────────────────────────────────────────────────────────
_IMAGE_TOKEN_ID: dict[str, int] = {
    # Qwen2.5-VL: <|image_pad|>
    "qwen2_5_vl":  151655,
    # LLaVA-1.6: <image> token id in LLaMA-based tokenisers
    "llava_1_6":   32000,
    # InternVL2: <img> placeholder
    "internvl2":   92543,
    # Phi-3-Vision: <|image_1|>
    "phi3_vision": 32044,
}

# Additional IDs that should *also* be masked (e.g. vision newline tokens)
_EXTRA_VISION_IDS: dict[str, list[int]] = {
    "qwen2_5_vl":  [151656],  # <|vision_start|>, <|vision_end|>
    "llava_1_6":   [],
    "internvl2":   [92544],
    "phi3_vision": [32045],
}


def build_vision_mask_from_labels(
    labels: "torch.Tensor",
) -> "torch.Tensor":
    """Build a text-token mask from a ``labels`` tensor (verl/TRL convention).

    Parameters
    ----------
    labels : Tensor, shape ``(B*G, L)``
        Token ids where positions that should **not** contribute to the loss
        are set to ``-100``.

    Returns
    -------
    Tensor, dtype uint8, shape ``(B*G, L)``
        ``1`` where ``labels != -100`` (response text tokens).
    """
    if not _HAS_TORCH:
        raise RuntimeError("build_vision_mask_from_labels requires PyTorch.")
    import torch as _torch
    return (labels != -100).to(_torch.uint8)


def build_vision_mask_from_input_ids(
    input_ids: "torch.Tensor",
    response_start_ids: Optional["torch.Tensor"] = None,
    image_token_id: Optional[int] = None,
    extra_mask_ids: Optional[list] = None,
    processor_type: str = "auto",
) -> "torch.Tensor":
    """Build a text-token mask from raw ``input_ids``.

    A token is marked as a **text / response** token (mask = 1) when:
      - it is NOT an image patch token (image_token_id), AND
      - it is NOT in ``extra_mask_ids``, AND
      - it occurs *after* the response start boundary (if provided).

    Parameters
    ----------
    input_ids : Tensor, shape ``(B*G, L)``
    response_start_ids : Tensor, shape ``(B*G,)``, optional
        Token index at which the model's generated response begins.
        Tokens before this index (system prompt + question + image patches)
        are excluded regardless of their token type.
    image_token_id : int, optional
        Override the image-patch token ID.  If ``None``, inferred from
        ``processor_type``.
    extra_mask_ids : list[int], optional
        Additional token IDs to mask (e.g. vision start/end markers).
    processor_type : str
        One of ``"auto"``, ``"qwen2_5_vl"``, ``"llava_1_6"``,
        ``"internvl2"``, ``"phi3_vision"``.

    Returns
    -------
    Tensor, dtype uint8, shape ``(B*G, L)``
    """
    if not _HAS_TORCH:
        raise RuntimeError("build_vision_mask_from_input_ids requires PyTorch.")
    import torch as _torch

    BG, L = input_ids.shape
    mask = _torch.ones(BG, L, dtype=_torch.uint8, device=input_ids.device)

    # Resolve image token ID
    if image_token_id is None and processor_type in _IMAGE_TOKEN_ID:
        image_token_id = _IMAGE_TOKEN_ID[processor_type]

    # Resolve extra IDs
    all_mask_ids: list[int] = list(extra_mask_ids or [])
    if processor_type in _EXTRA_VISION_IDS:
        all_mask_ids.extend(_EXTRA_VISION_IDS[processor_type])

    # Mask image patch tokens
    if image_token_id is not None:
        mask[input_ids == image_token_id] = 0

    # Mask extra token IDs
    for tid in all_mask_ids:
        mask[input_ids == tid] = 0

    # Mask positions before response start (prompt region)
    if response_start_ids is not None:
        positions = _torch.arange(L, device=input_ids.device).unsqueeze(0)  # (1, L)
        prompt_mask = positions < response_start_ids.unsqueeze(1)           # (BG, L)
        mask[prompt_mask] = 0

    return mask


def build_vision_mask_auto(
    processor_output: dict,
    processor_type: str = "auto",
    response_start_ids: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Dispatch to the correct vision-mask builder given a processor output dict.

    Supports the following processor output keys:
    - ``"input_ids"`` + ``"labels"`` (verl / TRL convention)
    - ``"input_ids"`` alone (heuristic based on processor_type)
    - ``"token_type_ids"`` (InternVL2 style)
    - ``"image_grid_thw"`` (Qwen2.5-VL: used to count patch tokens)

    Parameters
    ----------
    processor_output : dict
        Output from a HuggingFace processor / tokeniser.
    processor_type : str
        VLM family hint for image token ID lookup.
    response_start_ids : Tensor, optional
        Per-sequence response start token index.
    """
    if not _HAS_TORCH:
        raise RuntimeError("build_vision_mask_auto requires PyTorch.")
    import torch as _torch

    # Fast path: labels already mark non-loss positions with -100
    if "labels" in processor_output:
        labels = processor_output["labels"]
        if not isinstance(labels, _torch.Tensor):
            labels = _torch.tensor(labels)
        return build_vision_mask_from_labels(labels)

    if "input_ids" not in processor_output:
        raise ValueError(
            "processor_output must contain 'input_ids' or 'labels'."
        )

    input_ids = processor_output["input_ids"]
    if not isinstance(input_ids, _torch.Tensor):
        input_ids = _torch.tensor(input_ids)

    # InternVL2 provides explicit token_type_ids (0=text, 1=image)
    if "token_type_ids" in processor_output and processor_type in ("internvl2", "auto"):
        tti = processor_output["token_type_ids"]
        if not isinstance(tti, _torch.Tensor):
            tti = _torch.tensor(tti)
        mask = (tti != 1).to(_torch.uint8)
        if response_start_ids is not None:
            L = input_ids.shape[-1]
            positions = _torch.arange(L, device=mask.device).unsqueeze(0)
            prompt_mask = positions < response_start_ids.unsqueeze(1)
            mask[prompt_mask] = 0
        return mask

    # Generic: use image_token_id heuristic
    inferred_type = processor_type
    if inferred_type == "auto":
        # Try to infer from input_ids vocabulary range
        max_id = int(input_ids.max().item())
        if max_id > 150000:
            inferred_type = "qwen2_5_vl"
        elif max_id > 90000:
            inferred_type = "internvl2"
        elif max_id > 32040:
            inferred_type = "phi3_vision"
        else:
            inferred_type = "llava_1_6"

    return build_vision_mask_from_input_ids(
        input_ids,
        response_start_ids=response_start_ids,
        processor_type=inferred_type,
    )


def compact_vision_mask(
    vision_mask: "torch.Tensor",
) -> "torch.Tensor":
    """Ensure vision mask is contiguous uint8 on the same device as the input.

    Parameters
    ----------
    vision_mask : Tensor
        Any bool or integer tensor of shape ``(B*G, L)``.

    Returns
    -------
    Tensor, dtype uint8, contiguous.
    """
    if not _HAS_TORCH:
        raise RuntimeError("compact_vision_mask requires PyTorch.")
    import torch as _torch
    return vision_mask.to(_torch.uint8).contiguous()
