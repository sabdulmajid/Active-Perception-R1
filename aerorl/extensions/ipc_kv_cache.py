"""
aerorl/extensions/ipc_kv_cache.py
==================================
AeroRLSharedKVCache — zero-copy KV-cache hand-off between a vLLM rollout
process and a PyTorch training process via CUDA IPC.

Requires the compiled ``aerorl_ipc_ext`` C++/CUDA extension (built by
``setup.py``).  Falls back to reference-counted tensor storage when the
extension is not available (e.g., CPU-only development environments).

Usage (single-machine, two-process scenario)
---------------------------------------------
**Rollout process (vLLM side)**::

    from aerorl.extensions.ipc_kv_cache import AeroRLSharedKVCache

    cache = AeroRLSharedKVCache(num_layers=32, num_heads=32, head_dim=128)
    for layer_idx, (k, v) in enumerate(vllm_kv_pairs):
        cache.register_kv_layer(layer_idx, k, v)

    # Serialise handles and send to training process (e.g. via shared memory,
    # torch.distributed, or a simple socket).
    wire = cache.export_handles()          # bytes

**Training process (PyTorch side)**::

    from aerorl.extensions.ipc_kv_cache import AeroRLSharedKVCache

    cache = AeroRLSharedKVCache(num_layers=32, num_heads=32, head_dim=128)
    cache.import_handles(wire)             # bytes received from rollout proc

    k, v = cache.get_kv_layer(0)           # zero-copy Tensor view
"""

from __future__ import annotations

import logging
import pickle
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Try to load the compiled CUDA extension.
# ──────────────────────────────────────────────────────────────────────────
try:
    import aerorl_ipc_ext as _ipc_ext  # type: ignore[import]
    _HAS_IPC_EXT = True
    logger.debug("aerorl_ipc_ext loaded – zero-copy IPC enabled")
except ImportError:
    _ipc_ext = None  # type: ignore[assignment]
    _HAS_IPC_EXT = False
    logger.warning(
        "aerorl_ipc_ext not found.  Build it with `pip install -e .` or "
        "`python setup.py build_ext --inplace`.  "
        "Falling back to reference-copy KV sharing."
    )

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


# ──────────────────────────────────────────────────────────────────────────
# Wire-format helpers
# ──────────────────────────────────────────────────────────────────────────
class _KVLayerRecord:
    """Serialisable metadata for one KV-cache layer."""

    __slots__ = ("layer_idx", "k_handle", "v_handle", "k_shape", "v_shape",
                 "scalar_type", "device_index", "k_tensor", "v_tensor")

    def __init__(
        self,
        layer_idx: int,
        k_shape: List[int],
        v_shape: List[int],
        scalar_type: int,
        device_index: int,
        k_handle: Optional[bytes] = None,
        v_handle: Optional[bytes] = None,
        k_tensor=None,
        v_tensor=None,
    ):
        self.layer_idx = layer_idx
        self.k_handle = k_handle
        self.v_handle = v_handle
        self.k_shape = k_shape
        self.v_shape = v_shape
        self.scalar_type = scalar_type
        self.device_index = device_index
        self.k_tensor = k_tensor
        self.v_tensor = v_tensor


# ──────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────
class AeroRLSharedKVCache:
    """Zero-copy KV-cache shared between a vLLM rollout worker and a PyTorch
    training worker on the same GPU.

    Parameters
    ----------
    num_layers:
        Number of transformer layers (depth of the KV store).
    num_heads:
        Number of KV heads per layer.
    head_dim:
        Dimension of each head.
    dtype:
        ``torch.dtype`` for the KV tensors (default: ``torch.float16``).
    device:
        CUDA device index (default: ``0``).
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype=None,
        device: int = 0,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        if _HAS_TORCH:
            import torch as _torch
            self.dtype = dtype if dtype is not None else _torch.float16
        else:
            self.dtype = dtype  # store as-is for serialisation only

        # Layer records keyed by layer index
        self._records: Dict[int, _KVLayerRecord] = {}
        # Imported zero-copy tensors (training side)
        self._imported: Dict[int, Tuple] = {}

    # ── Rollout side (export) ───────────────────────────────────────────────

    def register_kv_layer(self, layer_idx: int, k_tensor, v_tensor) -> None:
        """Register KV tensors for one layer from the vLLM process.

        If the CUDA IPC extension is available, this exports raw IPC handles.
        Otherwise it keeps a strong reference to the tensors (fallback mode).

        Parameters
        ----------
        layer_idx:
            Zero-based layer index.
        k_tensor:
            Key tensor on CUDA, shape ``(batch, heads, seq, head_dim)`` or any
            contiguous layout used by the rollout engine.
        v_tensor:
            Value tensor with the same shape as ``k_tensor``.
        """
        if not _HAS_TORCH:
            raise RuntimeError(
                "register_kv_layer requires PyTorch to be installed."
            )
        import torch as _torch

        if not k_tensor.is_contiguous():
            k_tensor = k_tensor.contiguous()
        if not v_tensor.is_contiguous():
            v_tensor = v_tensor.contiguous()

        k_shape = list(k_tensor.shape)
        v_shape = list(v_tensor.shape)
        scalar_type = int(k_tensor.dtype)  # c10::ScalarType as int
        dev_idx = k_tensor.device.index if k_tensor.device.index is not None else 0

        if _HAS_IPC_EXT and _ipc_ext.is_ipc_supported():
            k_handle = _ipc_ext.export_ipc_handle(k_tensor)
            v_handle = _ipc_ext.export_ipc_handle(v_tensor)
            rec = _KVLayerRecord(
                layer_idx=layer_idx,
                k_shape=k_shape,
                v_shape=v_shape,
                scalar_type=scalar_type,
                device_index=dev_idx,
                k_handle=k_handle,
                v_handle=v_handle,
            )
        else:
            # Fallback: keep tensor references (single-process / no IPC)
            rec = _KVLayerRecord(
                layer_idx=layer_idx,
                k_shape=k_shape,
                v_shape=v_shape,
                scalar_type=scalar_type,
                device_index=dev_idx,
                k_tensor=k_tensor,
                v_tensor=v_tensor,
            )

        self._records[layer_idx] = rec

    def export_handles(self) -> bytes:
        """Serialise all registered IPC handles to ``bytes`` for IPC transport.

        Returns
        -------
        bytes
            Pickle payload containing a list of ``_KVLayerRecord`` objects
            (without tensor references — handles only).
        """
        exportable = []
        for rec in self._records.values():
            exportable.append({
                "layer_idx": rec.layer_idx,
                "k_handle": rec.k_handle,
                "v_handle": rec.v_handle,
                "k_shape": rec.k_shape,
                "v_shape": rec.v_shape,
                "scalar_type": rec.scalar_type,
                "device_index": rec.device_index,
            })
        return pickle.dumps(exportable)

    # ── Training side (import) ─────────────────────────────────────────────

    def import_handles(self, wire: bytes) -> None:
        """Reconstruct zero-copy Tensor views from serialised IPC handles.

        Must be called in the **training** process after ``export_handles()``
        is called in the rollout process and the payload is transmitted.

        Parameters
        ----------
        wire:
            Bytes produced by :meth:`export_handles`.
        """
        if not _HAS_TORCH:
            raise RuntimeError(
                "import_handles requires PyTorch to be installed."
            )

        records = pickle.loads(wire)
        for r in records:
            layer_idx = r["layer_idx"]
            if _HAS_IPC_EXT and r["k_handle"] is not None:
                k = _ipc_ext.import_ipc_handle(
                    r["k_handle"], r["k_shape"],
                    r["scalar_type"], r["device_index"]
                )
                v = _ipc_ext.import_ipc_handle(
                    r["v_handle"], r["v_shape"],
                    r["scalar_type"], r["device_index"]
                )
                self._imported[layer_idx] = (k, v)
            else:
                # Handles serialised but IPC ext not available in this proc;
                # or fallback mode (tensor refs copied during serialisation).
                logger.warning(
                    "Layer %d: IPC extension unavailable in import process; "
                    "no tensor available.  Register tensors directly via "
                    "register_kv_layer() in single-process mode.",
                    layer_idx,
                )

    def get_kv_layer(self, layer_idx: int):
        """Return the ``(key, value)`` tensor pair for *layer_idx*.

        In IPC mode the returned tensors are zero-copy views into the rollout
        process's GPU allocation.  In fallback mode they are the original
        tensors stored by ``register_kv_layer``.

        Returns
        -------
        Tuple[Tensor, Tensor] or Tuple[None, None]
        """
        if layer_idx in self._imported:
            return self._imported[layer_idx]
        if layer_idx in self._records:
            rec = self._records[layer_idx]
            if rec.k_tensor is not None:
                return rec.k_tensor, rec.v_tensor
        return None, None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def free(self) -> None:
        """Release all imported IPC tensors and registered references."""
        self._imported.clear()
        self._records.clear()

    def __del__(self) -> None:
        self.free()

    def __repr__(self) -> str:
        mode = "ipc" if (_HAS_IPC_EXT and self.num_layers > 0) else "fallback"
        return (
            f"AeroRLSharedKVCache(layers={self.num_layers}, "
            f"heads={self.num_heads}, head_dim={self.head_dim}, "
            f"registered={len(self._records)}, "
            f"imported={len(self._imported)}, mode={mode})"
        )

    # ── Convenience: single-process no-copy bridge ─────────────────────────

    @classmethod
    def from_kv_pairs(
        cls,
        kv_pairs,
        num_heads: int,
        head_dim: int,
        dtype=None,
        device: int = 0,
    ) -> "AeroRLSharedKVCache":
        """Construct and register all KV layers from an iterable of ``(k, v)`` pairs.

        Intended for single-process testing or frameworks that pass KV caches
        as Python objects rather than via OS IPC.

        Parameters
        ----------
        kv_pairs:
            Iterable of ``(key_tensor, value_tensor)`` in layer order.
        """
        pairs = list(kv_pairs)
        cache = cls(
            num_layers=len(pairs),
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
        for i, (k, v) in enumerate(pairs):
            cache.register_kv_layer(i, k, v)
        return cache
