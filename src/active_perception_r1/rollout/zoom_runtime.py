from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image

from active_perception_r1.envs.zoom_simulator import SimulatedZoomEnvironment
from active_perception_r1.utils.trace_parser import ZoomROIInvocation


STATUS_ZOOM_EXECUTED = "zoom_executed"
STATUS_INVALID_BBOX = "invalid_bbox"
STATUS_TOO_SMALL = "too_small"
STATUS_MALFORMED_ZOOM = "malformed_zoom"
STATUS_ZOOM_LIMIT_REACHED = "zoom_limit_reached"


def compose_view_bbox(
    parent_bbox_norm: tuple[float, float, float, float],
    local_bbox_norm: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    parent_x0, parent_y0, parent_x1, parent_y1 = parent_bbox_norm
    local_x0, local_y0, local_x1, local_y1 = local_bbox_norm
    parent_width = parent_x1 - parent_x0
    parent_height = parent_y1 - parent_y0
    return (
        parent_x0 + (local_x0 * parent_width),
        parent_y0 + (local_y0 * parent_height),
        parent_x0 + (local_x1 * parent_width),
        parent_y0 + (local_y1 * parent_height),
    )


def crop_image_to_normalized_bbox(
    image: Image.Image,
    bbox_norm: tuple[float, float, float, float],
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = bbox_norm
    left = int(round(x0 * image.width))
    top = int(round(y0 * image.height))
    right = int(round(x1 * image.width))
    bottom = int(round(y1 * image.height))
    right = max(left + 1, right)
    bottom = max(top + 1, bottom)
    crop_box = (left, top, min(image.width, right), min(image.height, bottom))
    return image.crop(crop_box), crop_box


@dataclass(frozen=True)
class ZoomExecutionTrace:
    status: str
    step_index: int
    current_view_bbox_norm: tuple[float, float, float, float]
    requested_bbox_norm: tuple[float, float, float, float] | None
    executed_bbox_norm: tuple[float, float, float, float] | None
    crop_bbox_pixels: tuple[int, int, int, int] | None
    area: float
    matched_region: str | None
    coverage: float
    iou: float
    weighted_signal: float
    observation_token: str | None
    tool_reward: float
    overscan: bool
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_error_token(code: str, detail: str) -> str:
    return f'<tool_error code="{code}" detail="{detail}" />'


def _weights() -> tuple[float, float, float, float]:
    from active_perception_r1.rewards.active_vision_reward import RewardWeights

    weights = RewardWeights()
    return (
        float(weights.valid_zoom_bonus),
        float(weights.out_of_bounds_penalty),
        float(weights.random_zoom_penalty),
        float(weights.overscan_penalty),
    )


def execute_zoom_action(
    *,
    image: Image.Image,
    current_view_bbox_norm: tuple[float, float, float, float],
    zoom_call: ZoomROIInvocation,
    extra_info: dict[str, Any],
    min_relative_area: float,
    max_relative_area: float,
) -> tuple[ZoomExecutionTrace, Image.Image | None]:
    valid_bonus, out_of_bounds_penalty, random_zoom_penalty, overscan_penalty = _weights()

    if not zoom_call.is_well_formed():
        trace = ZoomExecutionTrace(
            status=STATUS_INVALID_BBOX,
            step_index=zoom_call.step_index,
            current_view_bbox_norm=current_view_bbox_norm,
            requested_bbox_norm=None,
            executed_bbox_norm=None,
            crop_bbox_pixels=None,
            area=0.0,
            matched_region=None,
            coverage=0.0,
            iou=0.0,
            weighted_signal=0.0,
            observation_token=_build_error_token("invalid_bbox", "zoom_roi coordinates are malformed or reversed"),
            tool_reward=out_of_bounds_penalty,
            overscan=False,
            error="zoom_roi coordinates are malformed or reversed",
        )
        return trace, None

    requested_bbox_norm = zoom_call.to_normalized_bbox(image.width, image.height)
    x0, y0, x1, y1 = requested_bbox_norm
    if min(x0, y0, x1, y1) < 0.0 or max(x0, y0, x1, y1) > 1.0:
        trace = ZoomExecutionTrace(
            status=STATUS_INVALID_BBOX,
            step_index=zoom_call.step_index,
            current_view_bbox_norm=current_view_bbox_norm,
            requested_bbox_norm=requested_bbox_norm,
            executed_bbox_norm=None,
            crop_bbox_pixels=None,
            area=0.0,
            matched_region=None,
            coverage=0.0,
            iou=0.0,
            weighted_signal=0.0,
            observation_token=_build_error_token("out_of_bounds", "zoom_roi is outside the current image bounds"),
            tool_reward=out_of_bounds_penalty,
            overscan=False,
            error="zoom_roi is outside the current image bounds",
        )
        return trace, None

    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if area < min_relative_area:
        trace = ZoomExecutionTrace(
            status=STATUS_TOO_SMALL,
            step_index=zoom_call.step_index,
            current_view_bbox_norm=current_view_bbox_norm,
            requested_bbox_norm=requested_bbox_norm,
            executed_bbox_norm=None,
            crop_bbox_pixels=None,
            area=area,
            matched_region=None,
            coverage=0.0,
            iou=0.0,
            weighted_signal=0.0,
            observation_token=_build_error_token(
                "too_small",
                f"zoom_roi area {area:.4f} is below the minimum supported area {min_relative_area:.4f}",
            ),
            tool_reward=random_zoom_penalty,
            overscan=False,
            error=f"zoom_roi area {area:.4f} is below the minimum supported area {min_relative_area:.4f}",
        )
        return trace, None

    executed_bbox_norm = compose_view_bbox(current_view_bbox_norm, requested_bbox_norm)
    crop, crop_bbox_pixels = crop_image_to_normalized_bbox(image, requested_bbox_norm)

    simulation_extra_info = dict(extra_info)
    image_size = simulation_extra_info.get("image_size") or {}
    if "width" not in image_size or "height" not in image_size:
        simulation_extra_info["image_size"] = {
            "width": int(round(image.width / max(1e-6, current_view_bbox_norm[2] - current_view_bbox_norm[0]))),
            "height": int(round(image.height / max(1e-6, current_view_bbox_norm[3] - current_view_bbox_norm[1]))),
        }

    original_zoom_call = ZoomROIInvocation(
        step_index=zoom_call.step_index,
        raw_tag=zoom_call.raw_tag,
        x0=executed_bbox_norm[0],
        y0=executed_bbox_norm[1],
        x1=executed_bbox_norm[2],
        y1=executed_bbox_norm[3],
        normalized=True,
    )
    simulation = SimulatedZoomEnvironment.from_extra_info(simulation_extra_info).simulate_crop(original_zoom_call)
    overscan = area > max_relative_area
    tool_reward = valid_bonus + simulation.weighted_signal
    if overscan:
        tool_reward += overscan_penalty

    trace = ZoomExecutionTrace(
        status=STATUS_ZOOM_EXECUTED,
        step_index=zoom_call.step_index,
        current_view_bbox_norm=current_view_bbox_norm,
        requested_bbox_norm=requested_bbox_norm,
        executed_bbox_norm=executed_bbox_norm,
        crop_bbox_pixels=crop_bbox_pixels,
        area=area,
        matched_region=simulation.best_region_label,
        coverage=simulation.best_region_coverage,
        iou=simulation.best_region_iou,
        weighted_signal=simulation.weighted_signal,
        observation_token=simulation.observation_token,
        tool_reward=tool_reward,
        overscan=overscan,
        error=None,
    )
    return trace, crop


def malformed_zoom_trace(
    *,
    raw_tag: str,
    step_index: int,
    current_view_bbox_norm: tuple[float, float, float, float],
) -> ZoomExecutionTrace:
    return ZoomExecutionTrace(
        status=STATUS_MALFORMED_ZOOM,
        step_index=step_index,
        current_view_bbox_norm=current_view_bbox_norm,
        requested_bbox_norm=None,
        executed_bbox_norm=None,
        crop_bbox_pixels=None,
        area=0.0,
        matched_region=None,
        coverage=0.0,
        iou=0.0,
        weighted_signal=0.0,
        observation_token=_build_error_token("malformed_zoom", raw_tag),
        tool_reward=-0.20,
        overscan=False,
        error=raw_tag,
    )


def zoom_limit_trace(
    *,
    step_index: int,
    current_view_bbox_norm: tuple[float, float, float, float],
    max_zoom_calls: int,
) -> ZoomExecutionTrace:
    return ZoomExecutionTrace(
        status=STATUS_ZOOM_LIMIT_REACHED,
        step_index=step_index,
        current_view_bbox_norm=current_view_bbox_norm,
        requested_bbox_norm=None,
        executed_bbox_norm=None,
        crop_bbox_pixels=None,
        area=0.0,
        matched_region=None,
        coverage=0.0,
        iou=0.0,
        weighted_signal=0.0,
        observation_token=_build_error_token("zoom_limit_reached", f"maximum zoom calls reached ({max_zoom_calls})"),
        tool_reward=-0.10,
        overscan=False,
        error=f"maximum zoom calls reached ({max_zoom_calls})",
    )


def build_observation_message(
    trace: ZoomExecutionTrace,
    *,
    continue_instruction: str,
    image: Image.Image | None,
) -> dict[str, Any]:
    text = (
        "Tool observation:\n"
        f"{trace.observation_token or _build_error_token('empty_observation', 'no observation produced')}\n"
        f"{continue_instruction}"
    )
    if image is None:
        return {"role": "user", "content": text}
    return {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": text},
        ],
    }
