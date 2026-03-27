from __future__ import annotations

import unittest

from PIL import Image

from active_perception_r1.rollout import (
    STATUS_MALFORMED_ZOOM,
    STATUS_TOO_SMALL,
    STATUS_ZOOM_EXECUTED,
    STATUS_ZOOM_LIMIT_REACHED,
    compose_view_bbox,
    execute_zoom_action,
    malformed_zoom_trace,
    zoom_limit_trace,
)
from active_perception_r1.utils.trace_parser import ZoomROIInvocation


class ZoomRuntimeTests(unittest.TestCase):
    def assert_bbox_close(
        self,
        actual: tuple[float, float, float, float],
        expected: tuple[float, float, float, float],
    ) -> None:
        for actual_value, expected_value in zip(actual, expected, strict=True):
            self.assertAlmostEqual(actual_value, expected_value, places=6)

    def test_compose_view_bbox_composes_nested_zoom(self) -> None:
        composed = compose_view_bbox((0.2, 0.3, 0.8, 0.9), (0.5, 0.5, 0.75, 0.75))
        self.assert_bbox_close(composed, (0.5, 0.6, 0.65, 0.75))

    def test_execute_zoom_action_projects_back_to_original_image(self) -> None:
        image = Image.new("RGB", (600, 600), color="white")
        zoom_call = ZoomROIInvocation(
            step_index=1,
            raw_tag='<zoom_roi x0="0.5" y0="0.5" x1="0.75" y1="0.75" />',
            x0=0.5,
            y0=0.5,
            x1=0.75,
            y1=0.75,
            normalized=True,
        )
        trace, crop = execute_zoom_action(
            image=image,
            current_view_bbox_norm=(0.2, 0.3, 0.8, 0.9),
            zoom_call=zoom_call,
            extra_info={
                "image_size": {"width": 1000, "height": 1000},
                "relevant_regions": [{"label": "target", "bbox": [0.5, 0.6, 0.65, 0.75], "weight": 1.0}],
            },
            min_relative_area=0.02,
            max_relative_area=0.65,
        )

        self.assertEqual(trace.status, STATUS_ZOOM_EXECUTED)
        self.assert_bbox_close(trace.executed_bbox_norm, (0.5, 0.6, 0.65, 0.75))
        self.assertEqual(trace.crop_bbox_pixels, (300, 300, 450, 450))
        self.assertEqual(crop.size, (150, 150))
        self.assertEqual(trace.matched_region, "target")
        self.assertGreater(trace.tool_reward, 1.0)

    def test_execute_zoom_action_rejects_tiny_crop(self) -> None:
        image = Image.new("RGB", (256, 256), color="white")
        zoom_call = ZoomROIInvocation(
            step_index=1,
            raw_tag='<zoom_roi x0="0.01" y0="0.01" x1="0.02" y1="0.02" />',
            x0=0.01,
            y0=0.01,
            x1=0.02,
            y1=0.02,
            normalized=True,
        )
        trace, crop = execute_zoom_action(
            image=image,
            current_view_bbox_norm=(0.0, 0.0, 1.0, 1.0),
            zoom_call=zoom_call,
            extra_info={"image_size": {"width": 256, "height": 256}},
            min_relative_area=0.02,
            max_relative_area=0.65,
        )

        self.assertEqual(trace.status, STATUS_TOO_SMALL)
        self.assertIsNone(crop)
        self.assertLess(trace.tool_reward, 0.0)
        self.assertIn("too_small", trace.observation_token)

    def test_error_traces_produce_explicit_tokens(self) -> None:
        malformed = malformed_zoom_trace(
            raw_tag="<zoom_roi broken />",
            step_index=1,
            current_view_bbox_norm=(0.0, 0.0, 1.0, 1.0),
        )
        limited = zoom_limit_trace(
            step_index=2,
            current_view_bbox_norm=(0.2, 0.3, 0.8, 0.9),
            max_zoom_calls=3,
        )

        self.assertEqual(malformed.status, STATUS_MALFORMED_ZOOM)
        self.assertEqual(limited.status, STATUS_ZOOM_LIMIT_REACHED)
        self.assertIn("malformed_zoom", malformed.observation_token)
        self.assertIn("zoom_limit_reached", limited.observation_token)


if __name__ == "__main__":
    unittest.main()
