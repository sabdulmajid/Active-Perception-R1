"""
tests/test_aerorl.py
=====================
Unit tests for the AeroRL library.

These tests cover all pure-Python logic paths and the PyTorch fallback
implementations (no CUDA / Triton required).  They run with the same
``PYTHONPATH=src python3 -m unittest discover -s tests -v`` command used for
the existing active_perception_r1 tests.
"""

import math
import sys
import unittest


# ──────────────────────────────────────────────────────────────────────────
# Skip marker for tests that need PyTorch
# ──────────────────────────────────────────────────────────────────────────
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

requires_torch = unittest.skipUnless(_HAS_TORCH, "PyTorch not installed")


# ──────────────────────────────────────────────────────────────────────────
# 1. Config tests
# ──────────────────────────────────────────────────────────────────────────
class TestAeroRLConfig(unittest.TestCase):

    def test_default_construction(self):
        from aerorl.config import AeroRLConfig
        cfg = AeroRLConfig()
        self.assertEqual(cfg.loss_type, "grpo")
        self.assertEqual(cfg.epsilon, 0.2)
        self.assertEqual(cfg.beta, 0.01)
        self.assertEqual(cfg.quant_ref_bits, 8)
        self.assertTrue(cfg.zero_copy_kv)
        self.assertTrue(cfg.mask_vision_tokens)

    def test_custom_construction(self):
        from aerorl.config import AeroRLConfig
        cfg = AeroRLConfig(
            zero_copy_kv=False,
            loss_type="gspo",
            epsilon=0.1,
            beta=0.05,
            quant_ref_bits=4,
            dapo_variance_filter=True,
        )
        self.assertFalse(cfg.zero_copy_kv)
        self.assertEqual(cfg.loss_type, "gspo")
        self.assertAlmostEqual(cfg.epsilon, 0.1)
        self.assertAlmostEqual(cfg.beta, 0.05)
        self.assertEqual(cfg.quant_ref_bits, 4)
        self.assertTrue(cfg.dapo_variance_filter)

    def test_use_quant_ref_property(self):
        from aerorl.config import AeroRLConfig
        self.assertTrue(AeroRLConfig(quant_ref_bits=8).use_quant_ref)
        self.assertTrue(AeroRLConfig(quant_ref_bits=4).use_quant_ref)
        self.assertFalse(AeroRLConfig(quant_ref_bits=0).use_quant_ref)

    def test_invalid_quant_bits_raises(self):
        from aerorl.config import AeroRLConfig
        with self.assertRaises(ValueError):
            AeroRLConfig(quant_ref_bits=3)

    def test_invalid_epsilon_raises(self):
        from aerorl.config import AeroRLConfig
        with self.assertRaises(ValueError):
            AeroRLConfig(epsilon=0.0)
        with self.assertRaises(ValueError):
            AeroRLConfig(epsilon=1.5)

    def test_negative_beta_raises(self):
        from aerorl.config import AeroRLConfig
        with self.assertRaises(ValueError):
            AeroRLConfig(beta=-0.01)

    def test_to_dict_roundtrip(self):
        from aerorl.config import AeroRLConfig
        cfg = AeroRLConfig(loss_type="cispo", epsilon=0.15)
        d = cfg.to_dict()
        self.assertEqual(d["loss_type"], "cispo")
        self.assertAlmostEqual(d["epsilon"], 0.15)


# ──────────────────────────────────────────────────────────────────────────
# 2. Vision mask tests
# ──────────────────────────────────────────────────────────────────────────
class TestVisionMask(unittest.TestCase):

    @requires_torch
    def test_from_labels_basic(self):
        from aerorl.utils.vision_mask import build_vision_mask_from_labels

        labels = torch.tensor([[1, 2, -100, -100, 5]])
        mask   = build_vision_mask_from_labels(labels)
        expected = torch.tensor([[1, 1, 0, 0, 1]], dtype=torch.uint8)
        self.assertTrue(torch.equal(mask, expected))

    @requires_torch
    def test_from_labels_all_masked(self):
        from aerorl.utils.vision_mask import build_vision_mask_from_labels

        labels = torch.full((2, 8), -100)
        mask   = build_vision_mask_from_labels(labels)
        self.assertEqual(mask.sum().item(), 0)

    @requires_torch
    def test_from_input_ids_qwen(self):
        from aerorl.utils.vision_mask import build_vision_mask_from_input_ids

        B, L = 2, 16
        ids  = torch.randint(0, 32000, (B, L))
        ids[:, :4] = 151655  # IMAGE_PAD_ID

        mask = build_vision_mask_from_input_ids(
            ids, processor_type="qwen2_5_vl"
        )
        # Image pad positions should be 0
        self.assertEqual(mask[:, :4].sum().item(), 0)
        # Other positions should be 1
        self.assertTrue(mask[:, 4:].all())

    @requires_torch
    def test_from_input_ids_response_start(self):
        from aerorl.utils.vision_mask import build_vision_mask_from_input_ids

        B, L = 2, 16
        ids  = torch.randint(0, 32000, (B, L))
        # Response starts at position 8
        response_start = torch.tensor([8, 8])
        mask = build_vision_mask_from_input_ids(
            ids,
            response_start_ids=response_start,
            processor_type="llava_1_6",
        )
        # Positions 0-7 should be masked
        self.assertEqual(mask[:, :8].sum().item(), 0)
        # Positions 8+ should be 1 (no image token ids in this synthetic data)
        self.assertEqual(mask[:, 8:].sum().item(), B * (L - 8))

    @requires_torch
    def test_compact_vision_mask(self):
        from aerorl.utils.vision_mask import compact_vision_mask

        mask = torch.tensor([[True, False, True]], dtype=torch.bool)
        out  = compact_vision_mask(mask)
        self.assertEqual(out.dtype, torch.uint8)
        self.assertTrue(out.is_contiguous())
        self.assertTrue(torch.equal(out, torch.tensor([[1, 0, 1]], dtype=torch.uint8)))


# ──────────────────────────────────────────────────────────────────────────
# 3. VisionMaskBuilder tests
# ──────────────────────────────────────────────────────────────────────────
class TestVisionMaskBuilder(unittest.TestCase):

    def test_detect_family_qwen(self):
        from aerorl.utils.processor_utils import detect_family
        self.assertEqual(detect_family("Qwen/Qwen2.5-VL-7B-Instruct"), "qwen2_5_vl")
        self.assertEqual(detect_family("qwen2_5_vl-3b"), "qwen2_5_vl")

    def test_detect_family_llava(self):
        from aerorl.utils.processor_utils import detect_family
        self.assertEqual(detect_family("llava-hf/llava-v1.6-mistral-7b-hf"), "llava_1_6")

    def test_detect_family_internvl(self):
        from aerorl.utils.processor_utils import detect_family
        self.assertEqual(detect_family("OpenGVLab/InternVL2-8B"), "internvl2")

    def test_detect_family_phi3(self):
        from aerorl.utils.processor_utils import detect_family
        self.assertEqual(detect_family("microsoft/Phi-3-vision-128k-instruct"), "phi3_vision")

    def test_detect_family_unknown(self):
        from aerorl.utils.processor_utils import detect_family
        self.assertEqual(detect_family("some-random-model"), "unknown")

    def test_from_model_constructor(self):
        from aerorl.utils.processor_utils import VisionMaskBuilder
        b = VisionMaskBuilder.from_model("Qwen/Qwen2.5-VL-7B-Instruct")
        self.assertEqual(b.family, "qwen2_5_vl")

    @requires_torch
    def test_build_qwen_mask(self):
        from aerorl.utils.processor_utils import VisionMaskBuilder

        B, L = 2, 32
        ids  = torch.randint(0, 32000, (B, L))
        ids[:, :8] = 151655  # IMAGE_PAD_ID

        builder = VisionMaskBuilder(family="qwen2_5_vl")
        mask    = builder.build({"input_ids": ids})
        self.assertEqual(mask[:, :8].sum().item(), 0)
        self.assertEqual(mask[:, 8:].sum().item(), B * (L - 8))

    @requires_torch
    def test_build_labels_fast_path(self):
        from aerorl.utils.processor_utils import VisionMaskBuilder

        labels  = torch.tensor([[1, 2, -100, 4]])
        builder = VisionMaskBuilder(family="qwen2_5_vl")
        mask    = builder.build({"labels": labels})
        expected = torch.tensor([[1, 1, 0, 1]], dtype=torch.uint8)
        self.assertTrue(torch.equal(mask, expected))

    @requires_torch
    def test_build_internvl_token_type_ids(self):
        from aerorl.utils.processor_utils import VisionMaskBuilder

        B, L = 2, 16
        ids = torch.randint(0, 32000, (B, L))
        tti = torch.zeros(B, L, dtype=torch.long)
        tti[:, :6] = 1  # image tokens

        builder = VisionMaskBuilder(family="internvl2")
        mask    = builder.build({"input_ids": ids, "token_type_ids": tti})
        self.assertEqual(mask[:, :6].sum().item(), 0)
        self.assertEqual(mask[:, 6:].sum().item(), B * (L - 6))

    @requires_torch
    def test_unknown_family_all_text(self):
        from aerorl.utils.processor_utils import VisionMaskBuilder
        import warnings

        B, L = 2, 8
        ids     = torch.randint(0, 32000, (B, L))
        builder = VisionMaskBuilder(family="unknown")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            mask = builder.build({"input_ids": ids})

        # Unknown family: all tokens treated as text
        self.assertEqual(mask.sum().item(), B * L)


# ──────────────────────────────────────────────────────────────────────────
# 4. GRPO loss tests
# ──────────────────────────────────────────────────────────────────────────
class TestGRPOLoss(unittest.TestCase):

    @requires_torch
    def _make_batch(self, BG=4, L=16, seed=0):
        torch.manual_seed(seed)
        pol_lp = torch.randn(BG, L) - 5.0
        old_lp = pol_lp.detach().clone()
        ref_lp = pol_lp.detach().clone() - 0.1
        adv    = torch.randn(BG)
        vm     = torch.ones(BG, L, dtype=torch.uint8)
        vm[:, :4] = 0  # first 4 are vision tokens
        sl     = torch.full((BG,), L, dtype=torch.int32)
        return pol_lp, old_lp, ref_lp, adv, vm, sl

    @requires_torch
    def test_grpo_scalar_finite(self):
        from aerorl.kernels.grpo_loss import grpo_loss

        pol_lp, old_lp, ref_lp, adv, vm, sl = self._make_batch()
        pol_lp = pol_lp.requires_grad_(True)
        loss = grpo_loss(pol_lp, old_lp, ref_lp, adv, vm, sl)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(math.isfinite(loss.item()))

    @requires_torch
    def test_grpo_backward(self):
        from aerorl.kernels.grpo_loss import grpo_loss

        pol_lp, old_lp, ref_lp, adv, vm, sl = self._make_batch()
        pol_lp = pol_lp.requires_grad_(True)
        loss = grpo_loss(pol_lp, old_lp, ref_lp, adv, vm, sl)
        loss.backward()
        self.assertIsNotNone(pol_lp.grad)
        self.assertFalse(torch.isnan(pol_lp.grad).any())

    @requires_torch
    def test_grpo_vision_mask_zeros_grad(self):
        """Gradient at vision-token positions must be zero."""
        from aerorl.kernels.grpo_loss import grpo_loss

        BG, L = 4, 16
        pol_lp = (torch.randn(BG, L) - 5.0).requires_grad_(True)
        old_lp = pol_lp.detach().clone()
        ref_lp = pol_lp.detach().clone()
        adv    = torch.randn(BG)
        vm     = torch.zeros(BG, L, dtype=torch.uint8)
        vm[:, 8:] = 1  # only second half is text
        sl     = torch.full((BG,), L, dtype=torch.int32)

        loss = grpo_loss(pol_lp, old_lp, ref_lp, adv, vm, sl)
        loss.backward()
        # First 8 positions are masked; gradient must be zero there
        self.assertTrue(torch.all(pol_lp.grad[:, :8] == 0.0))

    @requires_torch
    def test_grpo_zero_beta(self):
        """With beta=0 the KL term should not affect the loss numerically."""
        from aerorl.kernels.grpo_loss import grpo_loss, _grpo_loss_pytorch

        pol, old, ref, adv, vm, sl = self._make_batch(BG=2, L=8)
        pol = pol.requires_grad_(False)

        loss_full = _grpo_loss_pytorch(pol, old, ref, adv, vm, sl, beta=0.01)
        loss_zero = _grpo_loss_pytorch(pol, old, ref, adv, vm, sl, beta=0.0)

        # They should differ (KL != 0 in general)
        self.assertFalse(torch.isclose(loss_full, loss_zero).item())

    @requires_torch
    def test_grpo_all_masked_does_not_crash(self):
        """All-zero vision mask (no text tokens) should not produce NaN."""
        from aerorl.kernels.grpo_loss import grpo_loss

        BG, L = 2, 8
        pol = (torch.randn(BG, L) - 5.0).requires_grad_(True)
        old = pol.detach()
        ref = pol.detach()
        adv = torch.randn(BG)
        vm  = torch.zeros(BG, L, dtype=torch.uint8)  # no text tokens
        sl  = torch.full((BG,), L, dtype=torch.int32)

        loss = grpo_loss(pol, old, ref, adv, vm, sl)
        self.assertTrue(math.isfinite(loss.item()))


# ──────────────────────────────────────────────────────────────────────────
# 5. GSPO loss tests
# ──────────────────────────────────────────────────────────────────────────
class TestGSPOLoss(unittest.TestCase):

    @requires_torch
    def test_gspo_top_k_selection(self):
        """Only top-k sequences should receive gradient."""
        from aerorl.kernels.gspo_loss import _build_sparse_mask

        G   = 4
        adv = torch.tensor([0.1, 2.0, -0.5, -3.0])  # |adv| sorted: 3.0, 2.0, 0.5, 0.1
        mask = _build_sparse_mask(adv, G=G, top_k_ratio=0.5)
        # Top-2 by |adv|: indices 1 (2.0) and 3 (-3.0)
        self.assertEqual(mask.sum().item(), 2)
        self.assertEqual(mask[1].item(), 1)
        self.assertEqual(mask[3].item(), 1)

    @requires_torch
    def test_gspo_k1_keeps_one(self):
        from aerorl.kernels.gspo_loss import _build_sparse_mask

        G = 8
        adv = torch.randn(G)
        mask = _build_sparse_mask(adv, G=G, top_k_ratio=0.125)  # k=1
        self.assertEqual(mask.sum().item(), 1)

    @requires_torch
    def test_gspo_scalar_finite(self):
        from aerorl.kernels.gspo_loss import gspo_loss

        BG, L, G = 8, 16, 4
        torch.manual_seed(7)
        pol = (torch.randn(BG, L) - 5.0).requires_grad_(True)
        old = pol.detach()
        ref = pol.detach() - 0.1
        adv = torch.randn(BG)
        vm  = torch.ones(BG, L, dtype=torch.uint8)
        vm[:, :4] = 0
        sl  = torch.full((BG,), L, dtype=torch.int32)

        loss = gspo_loss(pol, old, ref, adv, vm, sl, G=G, top_k_ratio=0.5)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(math.isfinite(loss.item()))

    @requires_torch
    def test_gspo_backward(self):
        from aerorl.kernels.gspo_loss import gspo_loss

        BG, L, G = 4, 16, 4
        torch.manual_seed(11)
        pol = (torch.randn(BG, L) - 5.0).requires_grad_(True)
        old = pol.detach()
        ref = pol.detach()
        adv = torch.randn(BG)
        vm  = torch.ones(BG, L, dtype=torch.uint8)
        sl  = torch.full((BG,), L, dtype=torch.int32)

        loss = gspo_loss(pol, old, ref, adv, vm, sl, G=G, top_k_ratio=0.5)
        loss.backward()
        self.assertIsNotNone(pol.grad)
        self.assertFalse(torch.isnan(pol.grad).any())


# ──────────────────────────────────────────────────────────────────────────
# 6. CISPO loss tests
# ──────────────────────────────────────────────────────────────────────────
class TestCISPOLoss(unittest.TestCase):

    @requires_torch
    def test_cispo_scalar_finite(self):
        from aerorl.kernels.cispo_loss import cispo_loss

        BG, L = 4, 16
        torch.manual_seed(42)
        pol = (torch.randn(BG, L) - 5.0).requires_grad_(True)
        old = pol.detach()
        ref = pol.detach() - 0.1
        adv = torch.randn(BG)
        vm  = torch.ones(BG, L, dtype=torch.uint8)
        vm[:, :4] = 0
        sl  = torch.full((BG,), L, dtype=torch.int32)

        loss = cispo_loss(pol, old, ref, adv, vm, sl)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(math.isfinite(loss.item()))

    @requires_torch
    def test_cispo_backward(self):
        from aerorl.kernels.cispo_loss import cispo_loss

        BG, L = 4, 16
        torch.manual_seed(99)
        pol = (torch.randn(BG, L) - 5.0).requires_grad_(True)
        old = pol.detach()
        ref = pol.detach()
        adv = torch.randn(BG)
        vm  = torch.ones(BG, L, dtype=torch.uint8)
        sl  = torch.full((BG,), L, dtype=torch.int32)

        loss = cispo_loss(pol, old, ref, adv, vm, sl)
        loss.backward()
        self.assertIsNotNone(pol.grad)
        self.assertFalse(torch.isnan(pol.grad).any())

    @requires_torch
    def test_cispo_sequence_ratio_clipping(self):
        """When policy = old, seq_ratio = 1.0, loss should be near zero surrogate."""
        from aerorl.kernels.cispo_loss import _cispo_loss_pytorch

        BG, L = 2, 8
        pol = torch.zeros(BG, L)  # log-probs all zero
        old = torch.zeros(BG, L)  # same
        ref = torch.zeros(BG, L)
        adv = torch.zeros(BG)     # zero advantages → zero surrogate
        vm  = torch.ones(BG, L, dtype=torch.uint8)
        sl  = torch.full((BG,), L, dtype=torch.int32)

        loss = _cispo_loss_pytorch(pol, old, ref, adv, vm, sl, epsilon_seq=0.2, beta=0.0)
        # With zero advantages and zero ratio, loss should be 0
        self.assertAlmostEqual(loss.item(), 0.0, places=5)


# ──────────────────────────────────────────────────────────────────────────
# 7. DAPO filter tests
# ──────────────────────────────────────────────────────────────────────────
class TestDAPOFilter(unittest.TestCase):

    @requires_torch
    def test_high_variance_passes(self):
        from aerorl.kernels.dapo_filter import dapo_variance_mask

        # G=4: rewards spread → high variance → all pass
        rewards = torch.tensor([1.0, -1.0, 2.0, -2.0])
        mask = dapo_variance_mask(rewards, G=4, min_variance=1e-4)
        self.assertEqual(mask.sum().item(), 4)

    @requires_torch
    def test_low_variance_filtered(self):
        from aerorl.kernels.dapo_filter import dapo_variance_mask

        # All same reward → zero variance → filtered
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
        mask = dapo_variance_mask(rewards, G=4, min_variance=1e-4)
        self.assertEqual(mask.sum().item(), 0)

    @requires_torch
    def test_mixed_groups(self):
        from aerorl.kernels.dapo_filter import dapo_variance_mask

        # B=2, G=4: first group diverse, second uniform
        rewards = torch.tensor([1.0, -1.0, 2.0, -2.0,  # diverse (passes)
                                 0.5,  0.5, 0.5,  0.5])  # uniform (fails)
        mask = dapo_variance_mask(rewards, G=4, min_variance=1e-4)
        self.assertEqual(mask[:4].sum().item(), 4)
        self.assertEqual(mask[4:].sum().item(), 0)

    @requires_torch
    def test_apply_dapo_filter_returns_three(self):
        from aerorl.kernels.dapo_filter import apply_dapo_filter

        rewards = torch.tensor([1.0, -1.0, 2.0, -2.0])
        adv     = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp(min=1e-8)
        r2, a2, keep = apply_dapo_filter(rewards, adv, G=4, min_variance=1e-4)
        self.assertEqual(r2.shape, rewards.shape)
        self.assertEqual(a2.shape, adv.shape)
        self.assertEqual(keep.shape, rewards.shape)


# ──────────────────────────────────────────────────────────────────────────
# 8. Shared KV cache tests
# ──────────────────────────────────────────────────────────────────────────
class TestSharedKVCache(unittest.TestCase):

    def test_construction(self):
        from aerorl.extensions.ipc_kv_cache import AeroRLSharedKVCache
        cache = AeroRLSharedKVCache(num_layers=4, num_heads=8, head_dim=64)
        self.assertEqual(cache.num_layers, 4)
        self.assertEqual(cache.num_heads, 8)
        self.assertEqual(cache.head_dim, 64)

    @requires_torch
    def test_register_and_get(self):
        """Fallback mode: register tensors and retrieve without IPC."""
        from aerorl.extensions.ipc_kv_cache import AeroRLSharedKVCache

        cache = AeroRLSharedKVCache(num_layers=2, num_heads=4, head_dim=32)
        k0 = torch.randn(1, 4, 8, 32)
        v0 = torch.randn(1, 4, 8, 32)
        cache.register_kv_layer(0, k0, v0)

        k_ret, v_ret = cache.get_kv_layer(0)
        self.assertIsNotNone(k_ret)
        self.assertTrue(torch.equal(k_ret, k0))
        self.assertTrue(torch.equal(v_ret, v0))

    @requires_torch
    def test_get_missing_layer_returns_none(self):
        from aerorl.extensions.ipc_kv_cache import AeroRLSharedKVCache

        cache = AeroRLSharedKVCache(num_layers=2, num_heads=4, head_dim=32)
        k, v = cache.get_kv_layer(99)
        self.assertIsNone(k)
        self.assertIsNone(v)

    @requires_torch
    def test_from_kv_pairs(self):
        from aerorl.extensions.ipc_kv_cache import AeroRLSharedKVCache

        pairs = [
            (torch.randn(1, 4, 8, 32), torch.randn(1, 4, 8, 32))
            for _ in range(3)
        ]
        cache = AeroRLSharedKVCache.from_kv_pairs(pairs, num_heads=4, head_dim=32)
        self.assertEqual(cache.num_layers, 3)
        for i in range(3):
            k, v = cache.get_kv_layer(i)
            self.assertTrue(torch.equal(k, pairs[i][0]))

    @requires_torch
    def test_free_clears_records(self):
        from aerorl.extensions.ipc_kv_cache import AeroRLSharedKVCache

        cache = AeroRLSharedKVCache(num_layers=1, num_heads=4, head_dim=32)
        cache.register_kv_layer(0, torch.randn(1, 4, 8, 32), torch.randn(1, 4, 8, 32))
        cache.free()
        k, v = cache.get_kv_layer(0)
        self.assertIsNone(k)

    @requires_torch
    def test_repr(self):
        from aerorl.extensions.ipc_kv_cache import AeroRLSharedKVCache

        cache = AeroRLSharedKVCache(num_layers=4, num_heads=8, head_dim=64)
        r = repr(cache)
        self.assertIn("AeroRLSharedKVCache", r)


# ──────────────────────────────────────────────────────────────────────────
# 9. gather_log_probs_and_free test
# ──────────────────────────────────────────────────────────────────────────
class TestGatherLogProbs(unittest.TestCase):

    @requires_torch
    def test_gather_correct_shape(self):
        from aerorl.utils.quant_ref import gather_log_probs_and_free

        BG, L, V = 2, 8, 100
        logits  = torch.randn(BG, L, V)
        actions = torch.randint(0, V, (BG, L))
        out     = gather_log_probs_and_free(logits, actions)
        self.assertEqual(out.shape, torch.Size([BG, L]))

    @requires_torch
    def test_gather_vision_mask_zeros(self):
        from aerorl.utils.quant_ref import gather_log_probs_and_free

        BG, L, V = 2, 8, 100
        logits  = torch.randn(BG, L, V)
        actions = torch.randint(0, V, (BG, L))
        vm      = torch.ones(BG, L, dtype=torch.uint8)
        vm[:, :3] = 0

        out = gather_log_probs_and_free(logits, actions, vision_mask=vm)
        self.assertTrue(torch.all(out[:, :3] == 0.0))

    @requires_torch
    def test_gather_values_correct(self):
        """Manually verify gather produces log_softmax of the chosen token."""
        from aerorl.utils.quant_ref import gather_log_probs_and_free
        import torch.nn.functional as F

        BG, L, V = 1, 4, 10
        torch.manual_seed(0)
        logits  = torch.randn(BG, L, V)
        actions = torch.randint(0, V, (BG, L))

        expected = F.log_softmax(logits.float(), dim=-1).gather(
            -1, actions.unsqueeze(-1).long()
        ).squeeze(-1)

        out = gather_log_probs_and_free(logits.clone(), actions.clone())
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))


# ──────────────────────────────────────────────────────────────────────────
# 10. Public API smoke test
# ──────────────────────────────────────────────────────────────────────────
class TestPublicAPI(unittest.TestCase):

    def test_top_level_imports(self):
        import aerorl
        self.assertTrue(hasattr(aerorl, "AeroRLConfig"))
        self.assertTrue(hasattr(aerorl, "wrap_vlm_for_rl"))
        self.assertTrue(hasattr(aerorl, "AeroRLSharedKVCache"))
        self.assertTrue(hasattr(aerorl, "grpo_loss"))
        self.assertTrue(hasattr(aerorl, "gspo_loss"))
        self.assertTrue(hasattr(aerorl, "cispo_loss"))
        self.assertTrue(hasattr(aerorl, "VisionMaskBuilder"))
        self.assertTrue(hasattr(aerorl, "QuantisedRefModel"))
        self.assertTrue(hasattr(aerorl, "BackgroundRefHook"))
        self.assertTrue(hasattr(aerorl, "apply_dapo_filter"))

    def test_version_string(self):
        import aerorl
        self.assertIsInstance(aerorl.__version__, str)
        self.assertTrue(len(aerorl.__version__) > 0)

    @requires_torch
    def test_basic_usage_script(self):
        """Smoke-test the basic_usage.py example end-to-end."""
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            from aerorl.examples import basic_usage
            basic_usage.main()

        output = f.getvalue()
        self.assertIn("GRPO loss", output)
        self.assertIn("GSPO loss", output)
        self.assertIn("CISPO loss", output)
        self.assertIn("completed successfully", output)


# ──────────────────────────────────────────────────────────────────────────
# 11. GRPO numerical sanity: ratio = 1 → surrogate = -advantage
# ──────────────────────────────────────────────────────────────────────────
class TestGRPONumerical(unittest.TestCase):

    @requires_torch
    def test_ratio_one_surrogate(self):
        """When pol_lp == old_lp and advantage > 0, surrogate = -advantage."""
        from aerorl.kernels.grpo_loss import _grpo_loss_pytorch

        BG, L = 1, 4
        # All log-probs equal → ratio = 1.0 everywhere
        lp  = torch.zeros(BG, L)
        adv = torch.tensor([2.0])
        vm  = torch.ones(BG, L, dtype=torch.uint8)
        sl  = torch.full((BG,), L, dtype=torch.int32)

        # With beta=0, loss = -min(1*A, clip(1)*A) = -A  (for A>0, ratio=1)
        loss = _grpo_loss_pytorch(lp, lp, lp, adv, vm, sl, beta=0.0)
        self.assertAlmostEqual(loss.item(), -2.0, places=5)

    @requires_torch
    def test_symmetric_advantages_cancel(self):
        """Equal positive/negative advantages should produce near-zero mean."""
        from aerorl.kernels.grpo_loss import _grpo_loss_pytorch

        BG, L = 4, 8
        torch.manual_seed(0)
        lp  = torch.randn(BG, L)
        adv = torch.tensor([1.0, -1.0, 1.0, -1.0])
        vm  = torch.ones(BG, L, dtype=torch.uint8)
        sl  = torch.full((BG,), L, dtype=torch.int32)

        # Policy same as old → ratio=1, no clipping. With ±1 advantages:
        # -min(+1, +1) + -min(-1,-1) = -1 + 1 = 0 per pair → ~0 total
        loss = _grpo_loss_pytorch(lp, lp, lp, adv, vm, sl, beta=0.0)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
