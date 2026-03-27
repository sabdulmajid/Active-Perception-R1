from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from active_perception_r1.envs.zoom_simulator import SimulatedZoomEnvironment
from active_perception_r1.utils.trace_parser import TraceParseResult, parse_reasoning_trace

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
ANSWER_LINE_RE = re.compile(r"(?im)^answer\s*:\s*(.+)$")


def normalize_answer(answer: str) -> str:
    answer = re.sub(r"\s+", " ", answer).strip()
    answer = answer.strip("\"'`")
    answer = answer.rstrip(". ")
    return answer.casefold()


def extract_final_answer(solution_str: str) -> str:
    tagged = ANSWER_TAG_RE.findall(solution_str)
    if tagged:
        return tagged[-1].strip()

    answer_lines = ANSWER_LINE_RE.findall(solution_str)
    if answer_lines:
        return answer_lines[-1].strip()

    non_empty_lines = [line.strip() for line in solution_str.splitlines() if line.strip()]
    return non_empty_lines[-1] if non_empty_lines else ""


@dataclass(frozen=True)
class RewardWeights:
    outcome_correct: float = 1.0
    outcome_incorrect: float = 0.0
    process_reward_scale: float = 0.35
    valid_zoom_bonus: float = 0.10
    malformed_zoom_penalty: float = -0.20
    out_of_bounds_penalty: float = -0.20
    missing_zoom_penalty: float = -0.15
    random_zoom_penalty: float = -0.10
    overscan_penalty: float = -0.08


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _coerce_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            if item is None:
                continue
            text = str(item)
            if text.strip():
                return text
        return default
    return str(value)


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, dict):
                return item
    return {}


def _coerce_aliases(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_trace_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, dict)]
    return []


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _serialise_tokens(tokens: list[str]) -> str:
    return json.dumps(tokens, ensure_ascii=True)


def _build_augmented_context(parse_result: TraceParseResult, observation_tokens: list[str]) -> str:
    if not observation_tokens:
        return parse_result.think_text
    joined_tokens = "\n".join(observation_tokens)
    if parse_result.think_text:
        return f"{parse_result.think_text}\n{joined_tokens}"
    return joined_tokens


def _score_executed_trace(
    *,
    extra_info: dict[str, Any],
    weights: RewardWeights,
    requires_zoom: bool,
) -> dict[str, Any] | None:
    trace_entries = _coerce_trace_list(extra_info.get("active_tool_trace"))
    if not trace_entries:
        return None

    process_reward = 0.0
    valid_zoom_count = 0
    out_of_bounds_count = 0
    malformed_zoom_count = 0
    relevant_zoom_count = 0
    best_region_coverage = 0.0
    best_region_iou = 0.0
    best_region_label = "none"
    observation_tokens: list[str] = []

    for trace in trace_entries:
        status = _coerce_string(trace.get("status"))
        process_reward += _coerce_float(trace.get("tool_reward"), default=0.0)

        observation_token = _coerce_string(trace.get("observation_token"))
        if observation_token:
            observation_tokens.append(observation_token)

        if status == "zoom_executed":
            valid_zoom_count += 1
            coverage = _coerce_float(trace.get("coverage"), default=0.0)
            if coverage > 0.0:
                relevant_zoom_count += 1
            if coverage > best_region_coverage:
                best_region_coverage = coverage
                best_region_iou = _coerce_float(trace.get("iou"), default=0.0)
                best_region_label = _coerce_string(trace.get("matched_region"), default="none") or "none"
        elif status == "malformed_zoom":
            malformed_zoom_count += 1
        elif status == "invalid_bbox":
            out_of_bounds_count += 1

    if requires_zoom and valid_zoom_count == 0:
        process_reward += weights.missing_zoom_penalty

    fallback_tokens = _coerce_string_list(extra_info.get("executed_observation_tokens"))
    if not observation_tokens:
        observation_tokens = fallback_tokens

    return {
        "tool_trace_source": "executed",
        "process_reward": float(process_reward),
        "zoom_call_count": int(len(trace_entries)),
        "valid_zoom_count": int(valid_zoom_count),
        "relevant_zoom_count": int(relevant_zoom_count),
        "malformed_zoom_count": int(malformed_zoom_count),
        "out_of_bounds_zoom_count": int(out_of_bounds_count),
        "best_region_coverage": float(best_region_coverage),
        "best_region_iou": float(best_region_iou),
        "best_region_label": best_region_label,
        "observation_tokens": observation_tokens,
    }


def _score_parsed_trace(
    *,
    parse_result: TraceParseResult,
    env: SimulatedZoomEnvironment,
    weights: RewardWeights,
    requires_zoom: bool,
    max_zoom_calls: int,
    min_relative_area: float,
    max_relative_area: float,
) -> dict[str, Any]:
    process_reward = 0.0
    malformed_zoom_count = len(parse_result.errors)
    process_reward += malformed_zoom_count * weights.malformed_zoom_penalty

    valid_zoom_count = 0
    out_of_bounds_count = 0
    relevant_zoom_count = 0
    best_region_coverage = 0.0
    best_region_iou = 0.0
    best_region_label = "none"
    observation_tokens: list[str] = []

    for zoom_call in parse_result.zoom_calls[:max_zoom_calls]:
        if not zoom_call.is_well_formed():
            out_of_bounds_count += 1
            process_reward += weights.out_of_bounds_penalty
            continue

        bbox_norm = zoom_call.to_normalized_bbox(env.image_width, env.image_height)
        area = max(0.0, bbox_norm[2] - bbox_norm[0]) * max(0.0, bbox_norm[3] - bbox_norm[1])
        if area < min_relative_area:
            process_reward += weights.random_zoom_penalty
            continue
        if area > max_relative_area:
            process_reward += weights.overscan_penalty

        simulation = env.simulate_crop(zoom_call)
        observation_tokens.append(simulation.observation_token)
        process_reward += weights.valid_zoom_bonus
        valid_zoom_count += 1

        if simulation.best_region_coverage > 0:
            relevant_zoom_count += 1

        if simulation.best_region_coverage > best_region_coverage:
            best_region_coverage = simulation.best_region_coverage
            best_region_iou = simulation.best_region_iou
            best_region_label = simulation.best_region_label or "none"

        process_reward += simulation.weighted_signal

    extra_zoom_count = max(0, len(parse_result.zoom_calls) - max_zoom_calls)
    process_reward += extra_zoom_count * weights.random_zoom_penalty

    if requires_zoom and valid_zoom_count == 0:
        process_reward += weights.missing_zoom_penalty

    return {
        "tool_trace_source": "parsed",
        "process_reward": float(process_reward),
        "zoom_call_count": int(len(parse_result.zoom_calls)),
        "valid_zoom_count": int(valid_zoom_count),
        "relevant_zoom_count": int(relevant_zoom_count),
        "malformed_zoom_count": int(malformed_zoom_count),
        "out_of_bounds_zoom_count": int(out_of_bounds_count),
        "best_region_coverage": float(best_region_coverage),
        "best_region_iou": float(best_region_iou),
        "best_region_label": best_region_label,
        "observation_tokens": observation_tokens,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    process_reward_scale: float = 0.35,
    max_zoom_calls: int = 3,
    min_relative_area: float = 0.02,
    max_relative_area: float = 0.65,
) -> dict[str, Any]:
    """Compute an active-perception reward for verl.

    The current implementation is a research scaffold: it parses zoom calls from
    the reasoning trace, simulates the crop against task metadata, appends
    structured crop tokens to an augmented context string, and scores whether
    those tool calls appear targeted and useful.
    """

    del data_source

    extra_info = _coerce_dict(extra_info)
    weights = RewardWeights(process_reward_scale=float(process_reward_scale))
    solution_text = _coerce_string(solution_str)
    parse_result = parse_reasoning_trace(solution_text)
    env = SimulatedZoomEnvironment.from_extra_info(extra_info)

    ground_truth_text = _coerce_string(ground_truth)
    answer_aliases = [ground_truth_text, *_coerce_aliases(extra_info.get("answer_aliases"))]
    prediction = extract_final_answer(solution_text)
    normalized_prediction = normalize_answer(prediction)
    normalized_aliases = {normalize_answer(alias) for alias in answer_aliases if str(alias).strip()}
    outcome_correct = normalized_prediction in normalized_aliases
    outcome_reward = weights.outcome_correct if outcome_correct else weights.outcome_incorrect

    requires_zoom = _coerce_bool(extra_info.get("requires_zoom"), default=bool(env.relevant_regions))

    trace_metrics = _score_executed_trace(extra_info=extra_info, weights=weights, requires_zoom=requires_zoom)
    if trace_metrics is None:
        trace_metrics = _score_parsed_trace(
            parse_result=parse_result,
            env=env,
            weights=weights,
            requires_zoom=requires_zoom,
            max_zoom_calls=max_zoom_calls,
            min_relative_area=min_relative_area,
            max_relative_area=max_relative_area,
        )

    process_reward = trace_metrics["process_reward"]
    augmented_context = _build_augmented_context(parse_result, trace_metrics["observation_tokens"])
    total_score = outcome_reward + (process_reward * weights.process_reward_scale)

    return {
        "score": float(total_score),
        "acc": float(outcome_correct),
        "pred": prediction,
        "outcome_reward": float(outcome_reward),
        "process_reward": float(process_reward),
        "visual_perception_reward": float(process_reward * weights.process_reward_scale),
        "tool_trace_source": trace_metrics["tool_trace_source"],
        "zoom_call_count": int(trace_metrics["zoom_call_count"]),
        "valid_zoom_count": int(trace_metrics["valid_zoom_count"]),
        "relevant_zoom_count": int(trace_metrics["relevant_zoom_count"]),
        "malformed_zoom_count": int(trace_metrics["malformed_zoom_count"]),
        "out_of_bounds_zoom_count": int(trace_metrics["out_of_bounds_zoom_count"]),
        "best_region_coverage": float(trace_metrics["best_region_coverage"]),
        "best_region_iou": float(trace_metrics["best_region_iou"]),
        "best_region_label": trace_metrics["best_region_label"],
        "simulated_observation_tokens": _serialise_tokens(trace_metrics["observation_tokens"]),
        "augmented_context": augmented_context,
    }
