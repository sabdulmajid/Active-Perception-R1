from __future__ import annotations

import unittest

from active_perception_r1.utils.training_profiles import recommend_training_profile


class TrainingProfileTests(unittest.TestCase):
    def test_qwen2_vl_family_prefers_stable_profile(self) -> None:
        profile = recommend_training_profile("Qwen/Qwen2.5-VL-7B-Instruct")
        self.assertFalse(profile.use_fused_kernels)
        self.assertFalse(profile.actor_use_torch_compile)
        self.assertFalse(profile.actor_fsdp_use_torch_compile)
        self.assertFalse(profile.ref_use_torch_compile)
        self.assertFalse(profile.ref_fsdp_use_torch_compile)
        self.assertEqual(profile.rollout_gpu_memory_utilization, 0.45)

    def test_generic_model_keeps_default_compile_profile(self) -> None:
        profile = recommend_training_profile("meta-llama/Llama-3.2-11B-Vision-Instruct")
        self.assertFalse(profile.use_fused_kernels)
        self.assertTrue(profile.actor_use_torch_compile)
        self.assertTrue(profile.actor_fsdp_use_torch_compile)
        self.assertTrue(profile.ref_use_torch_compile)
        self.assertTrue(profile.ref_fsdp_use_torch_compile)
        self.assertEqual(profile.rollout_gpu_memory_utilization, 0.55)


if __name__ == "__main__":
    unittest.main()
