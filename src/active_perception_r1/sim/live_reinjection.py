from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PIL import Image

from active_perception_r1.utils.trace_parser import parse_reasoning_trace


@dataclass(frozen=True)
class LiveStepRecord:
    step_index: int
    response: str
    used_zoom: bool
    observation_token: str | None


@dataclass(frozen=True)
class LiveEpisodeResult:
    final_response: str
    steps: list[LiveStepRecord]
    used_zoom_count: int


def _crop_from_zoom(image: Image.Image, x0: float, y0: float, x1: float, y1: float) -> Image.Image:
    width, height = image.size
    px0 = max(0, min(width, int(round(x0 * width))))
    py0 = max(0, min(height, int(round(y0 * height))))
    px1 = max(px0 + 1, min(width, int(round(x1 * width))))
    py1 = max(py0 + 1, min(height, int(round(y1 * height))))
    return image.crop((px0, py0, px1, py1))


def run_live_reinjection_episode(
    *,
    image: Image.Image,
    task_text: str,
    generator: Callable[[list[Image.Image], str], str],
    max_steps: int = 3,
) -> LiveEpisodeResult:
    """Run an iterative zoom-crop-reinject episode.

    The `generator` function is expected to perform model inference and return a
    textual response. On each step, the first valid `<zoom_roi .../>` action is
    executed into a crop that is reinjected as an additional image input.
    """

    images: list[Image.Image] = [image]
    conversation = task_text
    steps: list[LiveStepRecord] = []

    for step in range(1, max_steps + 1):
        response = generator(images, conversation)
        parsed = parse_reasoning_trace(response)

        zoom_call = None
        for candidate in parsed.zoom_calls:
            if candidate.is_well_formed():
                zoom_call = candidate
                break

        if zoom_call is None and "<answer>" in response.lower() and "</answer>" in response.lower():
            steps.append(LiveStepRecord(step_index=step, response=response, used_zoom=False, observation_token=None))
            return LiveEpisodeResult(final_response=response, steps=steps, used_zoom_count=sum(int(s.used_zoom) for s in steps))

        if zoom_call is None:
            steps.append(LiveStepRecord(step_index=step, response=response, used_zoom=False, observation_token=None))
            return LiveEpisodeResult(final_response=response, steps=steps, used_zoom_count=sum(int(s.used_zoom) for s in steps))

        source_image = images[-1]
        x0, y0, x1, y1 = zoom_call.to_normalized_bbox(source_image.width, source_image.height)
        crop = _crop_from_zoom(source_image, x0, y0, x1, y1)
        images.append(crop)

        observation_token = (
            f'<image_crop step="{step}" x0="{x0:.4f}" y0="{y0:.4f}" '
            f'x1="{x1:.4f}" y1="{y1:.4f}" />'
        )
        steps.append(
            LiveStepRecord(
                step_index=step,
                response=response,
                used_zoom=True,
                observation_token=observation_token,
            )
        )

        conversation = (
            f"{task_text}\n\n"
            f"Previous reasoning output:\n{response}\n"
            f"Tool observation:\n{observation_token}\n"
            "Use the reinjected crop evidence and answer as <answer>...</answer>."
        )

    final_response = steps[-1].response if steps else ""
    return LiveEpisodeResult(final_response=final_response, steps=steps, used_zoom_count=sum(int(s.used_zoom) for s in steps))
