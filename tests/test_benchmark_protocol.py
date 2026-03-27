from __future__ import annotations

import unittest

from PIL import Image

from active_perception_r1.bench.protocol import STRICT_ZOOM_FAILURE_ANSWER, run_active_default, run_active_strict_zoom
from active_perception_r1.utils.preflight import find_busy_gpus, inspect_dependencies, parse_gpu_status_csv, require_idle_gpus


class BenchmarkProtocolTests(unittest.TestCase):
    def test_strict_zoom_fails_closed_without_fallback_answer(self) -> None:
        image = Image.new("RGB", (200, 100), color=(255, 255, 255))
        calls: list[tuple[int, str]] = []

        def fake_generator(images, prompt):
            calls.append((len(images), prompt))
            return "<answer>42</answer>"

        result = run_active_strict_zoom(
            image=image,
            generator=fake_generator,
            zoom_prompt="zoom",
            retry_prompt="retry",
            answer_prompt="answer",
        )

        self.assertEqual(result.final_response, STRICT_ZOOM_FAILURE_ANSWER)
        self.assertFalse(result.used_crop)
        self.assertFalse(result.strict_zoom_satisfied)
        self.assertEqual(result.tool_status, "answered_without_zoom")
        self.assertEqual(result.tool_retry_count, 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1], "zoom")
        self.assertEqual(calls[1][1], "retry")

    def test_strict_zoom_retry_can_recover(self) -> None:
        image = Image.new("RGB", (200, 100), color=(255, 255, 255))
        calls: list[tuple[int, str]] = []

        def fake_generator(images, prompt):
            calls.append((len(images), prompt))
            if len(calls) == 1:
                return "no zoom"
            if len(calls) == 2:
                return "<zoom_roi x0=\"0.1\" y0=\"0.2\" x1=\"0.9\" y1=\"0.8\" />"
            return "<answer>42</answer>"

        result = run_active_strict_zoom(
            image=image,
            generator=fake_generator,
            zoom_prompt="zoom",
            retry_prompt="retry",
            answer_prompt="answer",
        )

        self.assertTrue(result.used_crop)
        self.assertTrue(result.strict_zoom_satisfied)
        self.assertEqual(result.tool_status, "zoom_executed")
        self.assertEqual(result.tool_retry_count, 1)
        self.assertEqual(result.step_count, 3)
        self.assertEqual(result.final_response, "<answer>42</answer>")
        self.assertEqual(calls[2][0], 2)

    def test_default_active_reports_answer_without_zoom(self) -> None:
        image = Image.new("RGB", (200, 100), color=(255, 255, 255))

        def fake_generator(images, prompt):
            return "<answer>42</answer>"

        result = run_active_default(
            image=image,
            task_text="question",
            generator=fake_generator,
            max_steps=2,
        )

        self.assertFalse(result.used_crop)
        self.assertEqual(result.tool_status, "answered_without_zoom")


class PreflightTests(unittest.TestCase):
    def test_parse_gpu_status_csv(self) -> None:
        statuses = parse_gpu_status_csv(
            "0, GPU A, 100, 10, 0\n"
            "1, GPU B, 100, 4096, 50\n"
        )

        self.assertEqual(len(statuses), 2)
        self.assertEqual(statuses[1].index, 1)
        self.assertEqual(statuses[1].memory_used_mib, 4096)

    def test_find_busy_gpus_uses_memory_or_utilization_threshold(self) -> None:
        statuses = parse_gpu_status_csv(
            "0, GPU A, 100, 10, 0\n"
            "1, GPU B, 100, 50, 20\n"
        )

        busy = find_busy_gpus(statuses, max_memory_used_mib=32, max_utilization_pct=10)
        self.assertEqual([status.index for status in busy], [1])

    def test_require_idle_gpus_raises_when_not_enough_idle_capacity(self) -> None:
        statuses = parse_gpu_status_csv(
            "0, GPU A, 100, 3000, 0\n"
            "1, GPU B, 100, 4000, 50\n"
        )

        with self.assertRaises(RuntimeError):
            require_idle_gpus(
                purpose="benchmark",
                required_count=1,
                statuses=statuses,
                allow_busy=False,
                max_memory_used_mib=2048,
                max_utilization_pct=10,
            )

    def test_inspect_dependencies_reports_missing_modules(self) -> None:
        def fake_import(name: str):
            if name == "missing_mod":
                raise ModuleNotFoundError("missing_mod")

            class Module:
                __version__ = "1.0"

            return Module()

        statuses = inspect_dependencies(["ok_mod", "missing_mod"], import_fn=fake_import)
        self.assertTrue(statuses[0].ok)
        self.assertFalse(statuses[1].ok)


if __name__ == "__main__":
    unittest.main()
