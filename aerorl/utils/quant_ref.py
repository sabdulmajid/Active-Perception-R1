"""
aerorl/utils/quant_ref.py
==========================
Background quantised reference-model hook.

The reference model is the most VRAM-intensive *non-trainable* component in
on-policy VLM RL.  This module provides:

1. ``QuantisedRefModel`` — wraps a HuggingFace model and quantises it to
   INT8 (via ``bitsandbytes``) or FP8 (via ``torchao``) in-place, then
   computes KL-divergence log-probs *only for text tokens*, and immediately
   frees the logits tensor after gathering token-level log-probs.

2. ``BackgroundRefHook`` — runs the reference-model forward pass in a
   background CUDA stream so that training and reference inference overlap.

Savings
-------
- INT8 reference model ≈ half the VRAM of a BF16 reference model.
- FP8 reference model ≈ quarter VRAM (requires Hopper/Blackwell GPU).
- Immediate logits deallocation avoids a ``(B*G, L, V)`` float tensor from
  living in VRAM past the log-prob gather step.

Usage
-----
::

    from aerorl.utils.quant_ref import QuantisedRefModel

    ref = QuantisedRefModel.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        quant_bits=8,
        backend="bitsandbytes",
    )
    ref_log_probs = ref.get_log_probs(input_ids, attention_mask,
                                       actions, vision_mask)
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


# ──────────────────────────────────────────────────────────────────────────
# Backend availability checks
# ──────────────────────────────────────────────────────────────────────────
def _check_bitsandbytes() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except ImportError:
        return False


def _check_torchao() -> bool:
    try:
        import torchao  # noqa: F401
        return True
    except ImportError:
        return False


# ──────────────────────────────────────────────────────────────────────────
# Low-level: gather log-probs from logits and immediately free logits
# ──────────────────────────────────────────────────────────────────────────
def gather_log_probs_and_free(
    logits: "torch.Tensor",
    actions: "torch.Tensor",
    vision_mask: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Gather token log-probabilities for ``actions`` then delete ``logits``.

    Parameters
    ----------
    logits : Tensor, shape ``(B*G, L, V)``
        Raw (un-normalised) token logits.
    actions : Tensor, shape ``(B*G, L)``
        Token IDs that were sampled (the chosen action at each step).
    vision_mask : Tensor, shape ``(B*G, L)``, optional
        If provided, positions where ``vision_mask == 0`` are set to ``0.0``
        in the returned log-probs (no gradient contribution from image tokens).

    Returns
    -------
    Tensor, shape ``(B*G, L)``
        Log-probabilities for the chosen actions, with vision tokens zeroed.
        The input ``logits`` tensor is deleted from Python scope immediately.
    """
    if not _HAS_TORCH:
        raise RuntimeError("gather_log_probs_and_free requires PyTorch.")
    import torch as _torch
    import torch.nn.functional as F

    # Compute log-softmax over vocab dim (last dim), then gather.
    # We do this in float32 for numerical stability even if logits are bf16.
    log_probs = F.log_softmax(logits.float(), dim=-1)  # (BG, L, V)

    # Gather chosen-action log-probs
    gathered = log_probs.gather(
        dim=-1,
        index=actions.unsqueeze(-1).long()
    ).squeeze(-1)  # (BG, L)

    # Immediately free the (potentially large) logits and log_probs tensors
    del logits
    del log_probs
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()

    # Zero out vision / prompt tokens
    if vision_mask is not None:
        gathered = gathered * vision_mask.float()

    return gathered.to(_torch.float32)


# ──────────────────────────────────────────────────────────────────────────
# QuantisedRefModel
# ──────────────────────────────────────────────────────────────────────────
class QuantisedRefModel:
    """Quantised reference model for KL-divergence computation.

    The model is loaded in ``quant_bits``-bit precision and placed in eval
    mode.  All parameters are frozen (``requires_grad=False``).

    Parameters
    ----------
    model:
        An already-instantiated HuggingFace ``PreTrainedModel``.
    quant_bits:
        Quantisation bit-width: ``8`` (INT8) or ``0`` (no quantisation,
        kept in original dtype).
    """

    def __init__(self, model, quant_bits: int = 8):
        if not _HAS_TORCH:
            raise RuntimeError("QuantisedRefModel requires PyTorch.")
        self._model = model
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        self.quant_bits = quant_bits

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        quant_bits: int = 8,
        backend: str = "bitsandbytes",
        device_map: str = "auto",
        torch_dtype=None,
        **kwargs,
    ) -> "QuantisedRefModel":
        """Load and quantise a HuggingFace model.

        Parameters
        ----------
        model_name_or_path : str
            HuggingFace model ID or local path.
        quant_bits : int
            ``8`` for INT8, ``0`` for full precision.
        backend : str
            ``"bitsandbytes"`` or ``"torchao"``.
        device_map : str
            Passed to ``from_pretrained``.
        torch_dtype : optional
            dtype for non-quantised layers.
        **kwargs:
            Forwarded to ``AutoModelForCausalLM.from_pretrained``.
        """
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                "QuantisedRefModel.from_pretrained requires `transformers`."
            ) from exc

        if torch_dtype is None and _HAS_TORCH:
            import torch as _torch
            torch_dtype = _torch.bfloat16

        if quant_bits == 8 and backend == "bitsandbytes":
            if not _check_bitsandbytes():
                logger.warning(
                    "bitsandbytes not installed; loading reference model in "
                    "full precision.  Install with: pip install bitsandbytes"
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    **kwargs,
                )
            else:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    quantization_config=bnb_config,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    **kwargs,
                )
                logger.info("Reference model loaded in INT8 (bitsandbytes).")

        elif quant_bits == 8 and backend == "torchao":
            if not _check_torchao():
                raise RuntimeError(
                    "torchao is not installed.  "
                    "Install with: pip install torchao"
                )
            import torchao
            from torchao.quantization import quantize_, int8_weight_only
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                **kwargs,
            )
            quantize_(model, int8_weight_only())
            logger.info("Reference model quantised with torchao INT8.")

        elif quant_bits == 4 and backend == "bitsandbytes":
            if not _check_bitsandbytes():
                raise RuntimeError(
                    "bitsandbytes not installed.  "
                    "Install with: pip install bitsandbytes"
                )
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch_dtype,
                **kwargs,
            )
            logger.info("Reference model loaded in NF4 (bitsandbytes 4-bit).")

        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                **kwargs,
            )

        return cls(model, quant_bits=quant_bits)

    @torch.no_grad() if _HAS_TORCH else (lambda f: f)
    def get_log_probs(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor",
        actions: "torch.Tensor",
        vision_mask: Optional["torch.Tensor"] = None,
        **forward_kwargs,
    ) -> "torch.Tensor":
        """Run a forward pass and return per-token log-probs for text tokens.

        Logits are freed immediately after gathering log-probs to avoid
        holding ``(B*G, L, V)`` in VRAM.

        Parameters
        ----------
        input_ids : Tensor, shape ``(B*G, L)``
        attention_mask : Tensor, shape ``(B*G, L)``
        actions : Tensor, shape ``(B*G, L)``
            Sampled action tokens (used to index into logits).
        vision_mask : Tensor, shape ``(B*G, L)``, optional
            Text-token mask; image-patch positions are zeroed in output.
        **forward_kwargs:
            Extra kwargs forwarded to the model (e.g. ``pixel_values``).

        Returns
        -------
        Tensor, shape ``(B*G, L)``
            Log-probs for chosen actions; image/prompt positions zeroed.
        """
        if not _HAS_TORCH:
            raise RuntimeError("get_log_probs requires PyTorch.")

        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **forward_kwargs,
        )
        # outputs.logits: (BG, L, V)
        # Shift left by 1 for auto-regressive next-token prediction alignment
        logits  = outputs.logits[:, :-1, :]   # (BG, L-1, V)
        actions_shifted = actions[:, 1:]       # (BG, L-1)

        vm_shifted = None
        if vision_mask is not None:
            vm_shifted = vision_mask[:, 1:]

        return gather_log_probs_and_free(logits, actions_shifted, vm_shifted)

    def __repr__(self) -> str:
        bits = self.quant_bits if self.quant_bits > 0 else "fp"
        return f"QuantisedRefModel(bits={bits}, model={type(self._model).__name__})"


# ──────────────────────────────────────────────────────────────────────────
# BackgroundRefHook
# ──────────────────────────────────────────────────────────────────────────
class BackgroundRefHook:
    """Run ``QuantisedRefModel.get_log_probs`` on a background CUDA stream.

    This allows training and reference inference to overlap in time,
    hiding the latency of the (cheaper, quantised) reference forward pass
    behind the policy-model forward/backward.

    Usage
    -----
    ::

        hook = BackgroundRefHook(quant_ref_model)

        # Issue the reference pass asynchronously:
        hook.submit(input_ids, attention_mask, actions, vision_mask)

        # ... run policy forward + backward here ...

        # Sync and retrieve:
        ref_log_probs = hook.result()
    """

    def __init__(self, ref_model: "QuantisedRefModel"):
        if not _HAS_TORCH:
            raise RuntimeError("BackgroundRefHook requires PyTorch.")
        import torch as _torch

        self._ref = ref_model
        self._stream = (
            _torch.cuda.Stream()
            if _torch.cuda.is_available()
            else None
        )
        self._future = None

    def submit(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor",
        actions: "torch.Tensor",
        vision_mask: Optional["torch.Tensor"] = None,
        **forward_kwargs,
    ) -> None:
        """Queue a reference-model forward pass on the background stream."""
        if self._stream is None:
            # CPU-only fallback: run synchronously, cache result
            self._future = self._ref.get_log_probs(
                input_ids, attention_mask, actions, vision_mask,
                **forward_kwargs
            )
            return

        import torch as _torch

        stream = self._stream
        ref    = self._ref

        # Pin args to avoid them being modified before the stream runs
        _input_ids      = input_ids.detach()
        _attn_mask      = attention_mask.detach()
        _actions        = actions.detach()
        _vision_mask    = vision_mask.detach() if vision_mask is not None else None
        _fwd_kwargs     = {k: v.detach() if isinstance(v, _torch.Tensor) else v
                           for k, v in forward_kwargs.items()}

        result_holder: list = []

        def _run():
            with _torch.cuda.stream(stream):
                lp = ref.get_log_probs(
                    _input_ids, _attn_mask, _actions, _vision_mask,
                    **_fwd_kwargs
                )
                result_holder.append(lp)

        import threading
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        self._future = (t, result_holder, stream)

    def result(self) -> "torch.Tensor":
        """Block until the background reference pass completes and return result."""
        if not _HAS_TORCH:
            raise RuntimeError("BackgroundRefHook.result requires PyTorch.")

        import torch as _torch

        if self._future is None:
            raise RuntimeError("submit() must be called before result().")

        if isinstance(self._future, _torch.Tensor):
            # CPU-only synchronous path
            lp = self._future
            self._future = None
            return lp

        t, result_holder, stream = self._future
        t.join()
        _torch.cuda.current_stream().wait_stream(stream)
        self._future = None

        if not result_holder:
            raise RuntimeError("Background reference pass produced no output.")
        return result_holder[0]
