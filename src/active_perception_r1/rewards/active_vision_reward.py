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


def _serialise_tokens(tokens: list[str]) -> str:
    return json.dumps(tokens, ensure_ascii=True)


def _build_augmented_context(parse_result: TraceParseResult, observation_tokens: list[str]) -> str:
    if not observation_tokens:
        return parse_result.think_text
    joined_tokens = "\n".join(observation_tokens)
    if parse_result.think_text:
        return f"{parse_result.think_text}\n{joined_tokens}"
    return joined_tokens


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

    extra_info = extra_info or {}
    weights = RewardWeights(process_reward_scale=float(process_reward_scale))
    parse_result = parse_reasoning_trace(solution_str)
    env = SimulatedZoomEnvironment.from_extra_info(extra_info)

    answer_aliases = [ground_truth, *extra_info.get("answer_aliases", [])]
    prediction = extract_final_answer(solution_str)
    normalized_prediction = normalize_answer(prediction)
    normalized_aliases = {normalize_answer(alias) for alias in answer_aliases if str(alias).strip()}
    outcome_correct = normalized_prediction in normalized_aliases
    outcome_reward = weights.outcome_correct if outcome_correct else weights.outcome_incorrect

    requires_zoom = _coerce_bool(extra_info.get("requires_zoom"), default=bool(env.relevant_regions))

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

    augmented_context = _build_augmented_context(parse_result, observation_tokens)
    total_score = outcome_reward + (process_reward * weights.process_reward_scale)

    return {
        "score": float(total_score),
        "acc": float(outcome_correct),
        "pred": prediction,
        "outcome_reward": float(outcome_reward),
        "process_reward": float(process_reward),
        "visual_perception_reward": float(process_reward * weights.process_reward_scale),
        "zoom_call_count": int(len(parse_result.zoom_calls)),
        "valid_zoom_count": int(valid_zoom_count),
        "relevant_zoom_count": int(relevant_zoom_count),
        "malformed_zoom_count": int(malformed_zoom_count),
        "out_of_bounds_zoom_count": int(out_of_bounds_count),
        "best_region_coverage": float(best_region_coverage),
        "best_region_iou": float(best_region_iou),
        "best_region_label": best_region_label,
        "simulated_observation_tokens": _serialise_tokens(observation_tokens),
        "augmented_context": augmented_context,
    }

