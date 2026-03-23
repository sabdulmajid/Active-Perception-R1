from __future__ import annotations

import random
import unittest

from active_perception_r1.rewards.self_driving_reward import compute_score as compute_self_driving_score
from active_perception_r1.rewards.self_driving_reward import extract_action
from active_perception_r1.sim.self_driving_lab import CONDITIONS
from active_perception_r1.sim.self_driving_lab import POLICIES
from active_perception_r1.sim.self_driving_lab import TASKS
from active_perception_r1.sim.self_driving_lab import generate_policy_sweep_report
from active_perception_r1.sim.self_driving_lab import generate_scene


class SelfDrivingRewardTests(unittest.TestCase):
    def test_extracts_action_from_tagged_output(self) -> None:
        response = "<think>Need caution.</think>\n<action>brake</action>"
        self.assertEqual(extract_action(response), "brake")

    def test_awards_action_bonus_for_matching_alias(self) -> None:
        response = (
            "<think>"
            "<zoom_roi x0=\"0.44\" y0=\"0.06\" x1=\"0.58\" y1=\"0.24\" />"
            "</think>\n"
            "<answer>red</answer>\n"
            "<action>stop</action>"
        )
        extra_info = {
            "requires_zoom": True,
            "safety_critical": True,
            "expected_action": "brake",
            "action_aliases": "stop",
            "image_size": {"width": 1600, "height": 900},
            "relevant_regions": [{"label": "traffic_light", "bbox": [0.47, 0.08, 0.55, 0.22], "weight": 1.0}],
            "answer_aliases": ["red light"],
        }

        reward = compute_self_driving_score(
            "self_driving_active_perception_v0",
            response,
            "red",
            extra_info=extra_info,
        )

        self.assertEqual(reward["predicted_action"], "stop")
        self.assertEqual(reward["action_reward"], 0.75)
        self.assertEqual(reward["safety_penalty"], 0.0)
        self.assertGreater(reward["score"], 1.5)

    def test_applies_safety_penalty_for_wrong_action(self) -> None:
        response = (
            "<think>"
            "<zoom_roi x0=\"0.44\" y0=\"0.06\" x1=\"0.58\" y1=\"0.24\" />"
            "</think>\n"
            "<answer>red</answer>\n"
            "<action>accelerate</action>"
        )
        extra_info = {
            "requires_zoom": True,
            "safety_critical": True,
            "expected_action": "brake",
            "action_aliases": ["stop", "slow_down"],
            "image_size": {"width": 1600, "height": 900},
            "relevant_regions": [{"label": "traffic_light", "bbox": [0.47, 0.08, 0.55, 0.22], "weight": 1.0}],
        }

        reward = compute_self_driving_score(
            "self_driving_active_perception_v0",
            response,
            "red",
            extra_info=extra_info,
        )

        self.assertEqual(reward["predicted_action"], "accelerate")
        self.assertEqual(reward["action_reward"], 0.0)
        self.assertEqual(reward["safety_penalty"], -1.0)
        self.assertLess(reward["score"], 1.0)


class SelfDrivingLabTests(unittest.TestCase):
    def test_scene_ids_include_seed_to_avoid_collisions(self) -> None:
        task = TASKS[0]
        condition = CONDITIONS[0]

        scene_a = generate_scene(task, condition, random.Random(1), scene_index=0, seed=7)
        scene_b = generate_scene(task, condition, random.Random(1), scene_index=0, seed=11)

        self.assertNotEqual(scene_a.scene_id, scene_b.scene_id)
        self.assertTrue(scene_a.scene_id.startswith("seed_7:"))
        self.assertTrue(scene_b.scene_id.startswith("seed_11:"))

    def test_report_metadata_and_markdown_capture_run_config(self) -> None:
        report = generate_policy_sweep_report(scenes_per_combo=3, seeds=[1, 2])
        metadata = report["metadata"]
        markdown = report["markdown_summary"]
        overall_summary = report["overall_summary"]

        self.assertEqual(metadata["num_scene_instances"], 2 * len(TASKS) * len(CONDITIONS) * 3)
        self.assertEqual(metadata["num_policy_rollouts"], metadata["num_scene_instances"] * len(POLICIES))
        self.assertIn("Generated with `3` scenes per task-condition-seed across seeds `[1, 2]`.", markdown)
        self.assertIn("`oracle_roi` dominates overall", markdown)
        self.assertNotIn("becomes dominant", markdown)
        self.assertGreater(
            overall_summary["active_verify"]["grounded_accuracy"],
            overall_summary["passive_cot"]["grounded_accuracy"],
        )


if __name__ == "__main__":
    unittest.main()
