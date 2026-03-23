from __future__ import annotations

import unittest

from PIL import Image

from active_perception_r1.sim.live_reinjection import run_live_reinjection_episode


class LiveReinjectionTests(unittest.TestCase):
    def test_reinjects_crop_and_finishes_with_answer(self) -> None:
        image = Image.new("RGB", (200, 100), color=(255, 255, 255))
        calls: list[tuple[int, str]] = []

        def fake_generator(images, text):
            calls.append((len(images), text))
            if len(calls) == 1:
                return (
                    "<think>I should zoom first "
                    "<zoom_roi x0=\"0.5\" y0=\"0.1\" x1=\"0.9\" y1=\"0.6\" />"
                    "</think>"
                )
            return "<answer>42</answer>"

        result = run_live_reinjection_episode(
            image=image,
            task_text="Read the value",
            generator=fake_generator,
            max_steps=3,
        )

        self.assertEqual(result.final_response, "<answer>42</answer>")
        self.assertEqual(result.used_zoom_count, 1)
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(calls[0][0], 1)
        self.assertEqual(calls[1][0], 2)

    def test_stops_when_no_zoom_or_answer(self) -> None:
        image = Image.new("RGB", (200, 100), color=(255, 255, 255))

        def fake_generator(images, text):
            return "I cannot answer."

        result = run_live_reinjection_episode(
            image=image,
            task_text="Read the value",
            generator=fake_generator,
            max_steps=3,
        )

        self.assertEqual(result.used_zoom_count, 0)
        self.assertEqual(len(result.steps), 1)


if __name__ == "__main__":
    unittest.main()
