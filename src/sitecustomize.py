from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys


def _patch_attention_utils_module(module) -> None:
    from verl.utils.npu_flash_attn_utils import index_first_axis, pad_input, rearrange, unpad_input

    if getattr(module, "_active_perception_flash_attn_fallback", False):
        return

    original_get_attention_functions = module._get_attention_functions

    def _patched_get_attention_functions():
        try:
            return original_get_attention_functions()
        except ModuleNotFoundError as exc:
            if exc.name not in {"flash_attn", "flash_attn.bert_padding"}:
                raise

            module._index_first_axis = index_first_axis
            module._pad_input = pad_input
            module._rearrange = rearrange
            module._unpad_input = unpad_input
            return index_first_axis, pad_input, rearrange, unpad_input

    module._get_attention_functions = _patched_get_attention_functions
    module._active_perception_flash_attn_fallback = True


class _AttentionUtilsLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader: importlib.abc.Loader) -> None:
        self._wrapped_loader = wrapped_loader

    def create_module(self, spec):
        if hasattr(self._wrapped_loader, "create_module"):
            return self._wrapped_loader.create_module(spec)
        return None

    def exec_module(self, module) -> None:
        self._wrapped_loader.exec_module(module)
        _patch_attention_utils_module(module)


class _AttentionUtilsFinder(importlib.abc.MetaPathFinder):
    TARGET = "verl.utils.attention_utils"

    def find_spec(self, fullname, path, target=None):
        if fullname != self.TARGET:
            return None

        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec

        if isinstance(spec.loader, _AttentionUtilsLoader):
            return spec

        spec.loader = _AttentionUtilsLoader(spec.loader)
        return spec


if "verl.utils.attention_utils" in sys.modules:
    _patch_attention_utils_module(sys.modules["verl.utils.attention_utils"])
else:
    sys.meta_path.insert(0, _AttentionUtilsFinder())
