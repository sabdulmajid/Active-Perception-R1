from __future__ import annotations

import json
import unittest

from active_perception_r1.rewards.active_vision_reward import compute_score
from active_perception_r1.utils.trace_parser import parse_reasoning_trace


class TraceParserTests(unittest.TestCase):
    def test_parses_normalized_zoom_tags(self) -> None:
        result = parse_reasoning_trace(
            "<think>Need detail <zoom_roi x0=\"0.10\" y0=\"0.20\" x1=\"0.40\" y1=\"0.60\" /></think>"
        )
        self.assertEqual(len(result.zoom_calls), 1)
        self.assertEqual(result.zoom_calls[0].step_index, 1)
        self.assertTrue(result.zoom_calls[0].normalized)

    def test_parses_xywh_pixel_tags(self) -> None:
        result = parse_reasoning_trace(
            "<think><zoom_roi x=\"100\" y=\"50\" w=\"300\" h=\"250\" normalized=\"false\" /></think>"
        )
        self.assertEqual(len(result.zoom_calls), 1)
        self.assertFalse(result.zoom_calls[0].normalized)
        self.assertEqual(result.zoom_calls[0].x1, 400.0)
        self.assertEqual(result.zoom_calls[0].y1, 300.0)

    def test_collects_malformed_tags(self) -> None:
        result = parse_reasoning_trace("<think><zoom_roi x0=\"0.1\" y0=\"0.2\" /></think>")
        self.assertEqual(len(result.zoom_calls), 0)
        self.assertEqual(len(result.errors), 1)


class ActiveVisionRewardTests(unittest.TestCase):
    def test_rewards_targeted_zoom_and_correct_answer(self) -> None:
        response = (
            "<think>"
            "I need the inset value."
            "<zoom_roi x0=\"0.68\" y0=\"0.05\" x1=\"0.95\" y1=\"0.30\" />"
            "</think>\n"
            "<answer>42</answer>"
        )
        extra_info = {
            "requires_zoom": True,
            "image_size": {"width": 1600, "height": 1200},
            "relevant_regions": [{"label": "inset", "bbox": [0.7, 0.08, 0.94, 0.28], "weight": 1.0}],
        }

        reward = compute_score("active_perception_v0", response, "42", extra_info=extra_info)

        self.assertEqual(reward["acc"], 1.0)
        self.assertGreater(reward["score"], 1.0)
        self.assertEqual(reward["relevant_zoom_count"], 1)
        self.assertGreater(reward["best_region_coverage"], 0.8)
        self.assertIn("image_crop", reward["augmented_context"])

    def test_penalizes_missing_zoom_when_task_requires_it(self) -> None:
        response = "<think>I can probably answer directly.</think>\n<answer>42</answer>"
        extra_info = {
            "requires_zoom": True,
            "image_size": {"width": 1600, "height": 1200},
            "relevant_regions": [{"label": "inset", "bbox": [0.7, 0.08, 0.94, 0.28], "weight": 1.0}],
        }

        reward = compute_score("active_perception_v0", response, "0", extra_info=extra_info)

        self.assertEqual(reward["valid_zoom_count"], 0)
        self.assertLess(reward["visual_perception_reward"], 0.0)

    def test_penalizes_random_or_tiny_crop(self) -> None:
        response = (
            "<think>"
            "<zoom_roi x0=\"0.01\" y0=\"0.01\" x1=\"0.02\" y1=\"0.02\" />"
            "</think>\n"
            "<answer>0</answer>"
        )
        extra_info = {
            "requires_zoom": True,
            "image_size": {"width": 1600, "height": 1200},
            "relevant_regions": [{"label": "inset", "bbox": [0.7, 0.08, 0.94, 0.28], "weight": 1.0}],
        }

        reward = compute_score("active_perception_v0", response, "0", extra_info=extra_info)

        self.assertEqual(reward["valid_zoom_count"], 0)
        self.assertLess(reward["process_reward"], 0.0)

    def test_returns_serialized_observation_tokens(self) -> None:
        response = (
            "<think>"
            "<zoom_roi x0=\"0.68\" y0=\"0.05\" x1=\"0.95\" y1=\"0.30\" />"
            "</think>\n"
            "<answer>42</answer>"
        )
        extra_info = {
            "requires_zoom": True,
            "image_size": {"width": 1600, "height": 1200},
            "relevant_regions": [{"label": "inset", "bbox": [0.7, 0.08, 0.94, 0.28], "weight": 1.0}],
        }

        reward = compute_score("active_perception_v0", response, "42", extra_info=extra_info)
        tokens = json.loads(reward["simulated_observation_tokens"])
        self.assertEqual(len(tokens), 1)
        self.assertIn("matched_region", tokens[0])

    def test_handles_list_shaped_verl_payload_fields(self) -> None:
        response = [
            "<think><zoom_roi x0=\"0.68\" y0=\"0.05\" x1=\"0.95\" y1=\"0.30\" /></think>\n<answer>42</answer>"
        ]
        ground_truth = ["42"]
        extra_info = [
            {
                "requires_zoom": True,
                "image_size": {"width": 1600, "height": 1200},
                "relevant_regions": [{"label": "inset", "bbox": [0.7, 0.08, 0.94, 0.28], "weight": 1.0}],
                "answer_aliases": "forty two",
            }
        ]

        reward = compute_score("active_perception_v0", response, ground_truth, extra_info=extra_info)

        self.assertEqual(reward["acc"], 1.0)
        self.assertEqual(reward["valid_zoom_count"], 1)
        self.assertGreater(reward["best_region_coverage"], 0.8)

    def test_accepts_scalar_answer_aliases(self) -> None:
        response = "<think>No zoom.</think>\n<answer>forty two</answer>"
        extra_info = {
            "requires_zoom": False,
            "answer_aliases": "forty two",
        }

        reward = compute_score("active_perception_v0", response, "42", extra_info=extra_info)

        self.assertEqual(reward["acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
