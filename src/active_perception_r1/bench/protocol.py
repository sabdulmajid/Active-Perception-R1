from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PIL import Image

from active_perception_r1.sim.live_reinjection import run_live_reinjection_episode
from active_perception_r1.utils.trace_parser import parse_reasoning_trace

STRICT_ZOOM_FAILURE_ANSWER = "<answer>__zoom_tool_failed__</answer>"


@dataclass(frozen=True)
class ActiveRunResult:
    initial_response: str
    final_response: str
    scoring_trace: str
    used_crop: bool
    step_count: int
    tool_status: str
    tool_retry_count: int
    strict_zoom_satisfied: bool


def crop_from_bbox(image: Image.Image, bbox_norm: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    x0 = max(0, min(width, int(round(bbox_norm[0] * width))))
    y0 = max(0, min(height, int(round(bbox_norm[1] * height))))
    x1 = max(x0 + 1, min(width, int(round(bbox_norm[2] * width))))
    y1 = max(y0 + 1, min(height, int(round(bbox_norm[3] * height))))
    return image.crop((x0, y0, x1, y1))


def crop_from_zoom_trace(image: Image.Image, response: str) -> Image.Image | None:
    parsed = parse_reasoning_trace(response)
    for candidate in parsed.zoom_calls:
        if candidate.is_well_formed():
            x0, y0, x1, y1 = candidate.to_normalized_bbox(image.width, image.height)
            return crop_from_bbox(image, (x0, y0, x1, y1))
    return None


def _describe_tool_status(response: str) -> str:
    parsed = parse_reasoning_trace(response)
    if any(candidate.is_well_formed() for candidate in parsed.zoom_calls):
        return "zoom_executed"
    if parsed.errors:
        return "malformed_zoom"
    if parsed.zoom_calls:
        return "invalid_zoom"
    if "<answer>" in response.lower():
        return "answered_without_zoom"
    return "missing_zoom"


def run_active_default(
    *,
    image: Image.Image,
    task_text: str,
    generator: Callable[[list[Image.Image], str], str],
    max_steps: int = 3,
) -> ActiveRunResult:
    live_result = run_live_reinjection_episode(
        image=image,
        task_text=task_text,
        generator=generator,
        max_steps=max_steps,
    )
    initial_response = live_result.steps[0].response if live_result.steps else ""
    scoring_trace = "\n".join(step.response for step in live_result.steps)
    if not scoring_trace and live_result.final_response:
        scoring_trace = live_result.final_response

    if live_result.used_zoom_count > 0:
        tool_status = "zoom_executed"
    else:
        tool_status = _describe_tool_status(initial_response or live_result.final_response)

    return ActiveRunResult(
        initial_response=initial_response,
        final_response=live_result.final_response,
        scoring_trace=scoring_trace,
        used_crop=live_result.used_zoom_count > 0,
        step_count=len(live_result.steps),
        tool_status=tool_status,
        tool_retry_count=0,
        strict_zoom_satisfied=live_result.used_zoom_count > 0,
    )


def run_active_strict_zoom(
    *,
    image: Image.Image,
    generator: Callable[[list[Image.Image], str], str],
    zoom_prompt: str,
    retry_prompt: str,
    answer_prompt: str,
    max_retries: int = 1,
    failure_answer: str = STRICT_ZOOM_FAILURE_ANSWER,
) -> ActiveRunResult:
    zoom_responses: list[str] = []
    crop = None

    zoom_response = generator([image], zoom_prompt)
    zoom_responses.append(zoom_response)
    crop = crop_from_zoom_trace(image, zoom_response)

    while crop is None and len(zoom_responses) <= max_retries:
        retry_response = generator([image], retry_prompt)
        zoom_responses.append(retry_response)
        crop = crop_from_zoom_trace(image, retry_response)

    initial_response = zoom_responses[0] if zoom_responses else ""
    tool_retry_count = max(0, len(zoom_responses) - 1)

    if crop is None:
        scoring_trace = "\n".join([*zoom_responses, failure_answer])
        return ActiveRunResult(
            initial_response=initial_response,
            final_response=failure_answer,
            scoring_trace=scoring_trace,
            used_crop=False,
            step_count=len(zoom_responses),
            tool_status=_describe_tool_status(zoom_responses[-1] if zoom_responses else ""),
            tool_retry_count=tool_retry_count,
            strict_zoom_satisfied=False,
        )

    final_response = generator([image, crop], answer_prompt)
    scoring_trace = "\n".join([*zoom_responses, final_response])
    return ActiveRunResult(
        initial_response=initial_response,
        final_response=final_response,
        scoring_trace=scoring_trace,
        used_crop=True,
        step_count=len(zoom_responses) + 1,
        tool_status="zoom_executed",
        tool_retry_count=tool_retry_count,
        strict_zoom_satisfied=True,
    )
