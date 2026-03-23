from __future__ import annotations

import re
from dataclasses import dataclass

THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
ZOOM_TAG_RE = re.compile(r"<zoom_roi(?P<attrs>[^>]*)/?>", re.IGNORECASE)
ATTR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_\-]*)\s*=\s*(['\"])(.*?)\2")


@dataclass(frozen=True)
class ZoomROIInvocation:
    step_index: int
    raw_tag: str
    x0: float
    y0: float
    x1: float
    y1: float
    normalized: bool

    def is_well_formed(self) -> bool:
        return (
            self.x0 < self.x1
            and self.y0 < self.y1
            and self.x0 >= 0.0
            and self.y0 >= 0.0
            and ((self.normalized and self.x1 <= 1.0 and self.y1 <= 1.0) or not self.normalized)
        )

    def to_normalized_bbox(self, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        if self.normalized:
            return (self.x0, self.y0, self.x1, self.y1)
        return (
            self.x0 / image_width,
            self.y0 / image_height,
            self.x1 / image_width,
            self.y1 / image_height,
        )


@dataclass(frozen=True)
class TraceParseResult:
    think_text: str
    zoom_calls: list[ZoomROIInvocation]
    errors: list[str]


def _extract_think_text(solution_str: str) -> str:
    match = THINK_BLOCK_RE.search(solution_str)
    if match:
        return match.group(1).strip()
    return solution_str


def _parse_attrs(raw_tag: str) -> dict[str, str]:
    attrs = {}
    for key, _, value in ATTR_RE.findall(raw_tag):
        attrs[key.lower()] = value
    return attrs


def _float_attr(attrs: dict[str, str], key: str) -> float | None:
    value = attrs.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _build_zoom_call(step_index: int, raw_tag: str) -> ZoomROIInvocation | None:
    attrs = _parse_attrs(raw_tag)

    x0 = _float_attr(attrs, "x0")
    y0 = _float_attr(attrs, "y0")
    x1 = _float_attr(attrs, "x1")
    y1 = _float_attr(attrs, "y1")

    if None in {x0, y0, x1, y1}:
        x = _float_attr(attrs, "x")
        y = _float_attr(attrs, "y")
        width = _float_attr(attrs, "w")
        height = _float_attr(attrs, "h")
        if None in {x, y, width, height}:
            return None
        x0 = x
        y0 = y
        x1 = x + width
        y1 = y + height

    normalized_attr = attrs.get("normalized")
    normalized = True
    if normalized_attr is not None:
        normalized = normalized_attr.strip().lower() not in {"0", "false", "no"}
    elif max(x0, y0, x1, y1) > 1.0:
        normalized = False

    return ZoomROIInvocation(
        step_index=step_index,
        raw_tag=raw_tag,
        x0=float(x0),
        y0=float(y0),
        x1=float(x1),
        y1=float(y1),
        normalized=normalized,
    )


def parse_reasoning_trace(solution_str: str) -> TraceParseResult:
    think_text = _extract_think_text(solution_str)
    zoom_calls: list[ZoomROIInvocation] = []
    errors: list[str] = []

    for step_index, match in enumerate(ZOOM_TAG_RE.finditer(think_text), start=1):
        raw_tag = match.group(0)
        zoom_call = _build_zoom_call(step_index=step_index, raw_tag=raw_tag)
        if zoom_call is None:
            errors.append(raw_tag)
            continue
        zoom_calls.append(zoom_call)

    return TraceParseResult(think_text=think_text, zoom_calls=zoom_calls, errors=errors)

