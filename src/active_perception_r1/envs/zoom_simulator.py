from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from active_perception_r1.utils.trace_parser import ZoomROIInvocation


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _area(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def intersection_over_region(
    crop_bbox: tuple[float, float, float, float],
    target_bbox: tuple[float, float, float, float],
) -> float:
    x0 = max(crop_bbox[0], target_bbox[0])
    y0 = max(crop_bbox[1], target_bbox[1])
    x1 = min(crop_bbox[2], target_bbox[2])
    y1 = min(crop_bbox[3], target_bbox[3])
    intersection = _area((x0, y0, x1, y1))
    region_area = _area(target_bbox)
    if region_area <= 0:
        return 0.0
    return intersection / region_area


def intersection_over_union(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    x0 = max(bbox_a[0], bbox_b[0])
    y0 = max(bbox_a[1], bbox_b[1])
    x1 = min(bbox_a[2], bbox_b[2])
    y1 = min(bbox_a[3], bbox_b[3])
    intersection = _area((x0, y0, x1, y1))
    union = _area(bbox_a) + _area(bbox_b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


@dataclass(frozen=True)
class RelevantRegion:
    label: str
    bbox: tuple[float, float, float, float]
    weight: float = 1.0


@dataclass(frozen=True)
class CropSimulationResult:
    crop_bbox_norm: tuple[float, float, float, float]
    crop_bbox_pixels: tuple[int, int, int, int]
    best_region_label: str | None
    best_region_coverage: float
    best_region_iou: float
    weighted_signal: float
    observation_token: str


class SimulatedZoomEnvironment:
    """Simulates zoom/crop actions from structured task metadata."""

    def __init__(self, image_width: int, image_height: int, relevant_regions: list[RelevantRegion]) -> None:
        self.image_width = max(1, int(image_width))
        self.image_height = max(1, int(image_height))
        self.relevant_regions = relevant_regions

    @classmethod
    def from_extra_info(cls, extra_info: dict[str, Any] | None) -> "SimulatedZoomEnvironment":
        extra_info = extra_info or {}
        image_size = extra_info.get("image_size") or {}
        image_width = int(image_size.get("width", 1000))
        image_height = int(image_size.get("height", 1000))
        relevant_regions = []
        for item in extra_info.get("relevant_regions", []):
            bbox = item.get("bbox", [0.0, 0.0, 1.0, 1.0])
            relevant_regions.append(
                RelevantRegion(
                    label=str(item.get("label", "region")),
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    weight=float(item.get("weight", 1.0)),
                )
            )
        return cls(image_width=image_width, image_height=image_height, relevant_regions=relevant_regions)

    def simulate_crop(self, zoom_call: ZoomROIInvocation) -> CropSimulationResult:
        bbox_norm = zoom_call.to_normalized_bbox(self.image_width, self.image_height)
        x0, y0, x1, y1 = bbox_norm
        crop_bbox_pixels = (
            int(round(_clamp(x0, 0.0, 1.0) * self.image_width)),
            int(round(_clamp(y0, 0.0, 1.0) * self.image_height)),
            int(round(_clamp(x1, 0.0, 1.0) * self.image_width)),
            int(round(_clamp(y1, 0.0, 1.0) * self.image_height)),
        )

        best_label = None
        best_coverage = 0.0
        best_iou = 0.0
        best_weighted_signal = 0.0

        for region in self.relevant_regions:
            coverage = intersection_over_region(bbox_norm, region.bbox)
            iou = intersection_over_union(bbox_norm, region.bbox)
            weighted_signal = coverage * region.weight
            if weighted_signal > best_weighted_signal:
                best_label = region.label
                best_coverage = coverage
                best_iou = iou
                best_weighted_signal = weighted_signal

        observation_token = (
            f'<image_crop step="{zoom_call.step_index}" '
            f'x0="{bbox_norm[0]:.4f}" y0="{bbox_norm[1]:.4f}" '
            f'x1="{bbox_norm[2]:.4f}" y1="{bbox_norm[3]:.4f}" '
            f'matched_region="{best_label or "none"}" '
            f'coverage="{best_coverage:.4f}" iou="{best_iou:.4f}" />'
        )

        return CropSimulationResult(
            crop_bbox_norm=bbox_norm,
            crop_bbox_pixels=crop_bbox_pixels,
            best_region_label=best_label,
            best_region_coverage=best_coverage,
            best_region_iou=best_iou,
            weighted_signal=best_weighted_signal,
            observation_token=observation_token,
        )

