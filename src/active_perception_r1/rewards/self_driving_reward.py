from __future__ import annotations

import re
from typing import Any

from active_perception_r1.rewards.active_vision_reward import compute_score as compute_active_vision_score
from active_perception_r1.rewards.active_vision_reward import normalize_answer

ACTION_TAG_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
ACTION_LINE_RE = re.compile(r"(?im)^action\s*:\s*(.+)$")


def _coerce_action_aliases(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def extract_action(solution_str: str) -> str:
    tagged = ACTION_TAG_RE.findall(solution_str)
    if tagged:
        return tagged[-1].strip()

    action_lines = ACTION_LINE_RE.findall(solution_str)
    if action_lines:
        return action_lines[-1].strip()

    return ""


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Self-driving reward wrapper around the generic active-perception reward."""

    extra_info = extra_info or {}
    base_result = compute_active_vision_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )

    expected_actions = [extra_info.get("expected_action"), *_coerce_action_aliases(extra_info.get("action_aliases"))]
    expected_actions = [normalize_answer(str(action)) for action in expected_actions if str(action).strip()]
    predicted_action = extract_action(solution_str)
    normalized_predicted_action = normalize_answer(predicted_action)

    action_match = bool(expected_actions) and normalized_predicted_action in set(expected_actions)
    action_reward = 0.75 if action_match else 0.0

    safety_critical = bool(extra_info.get("safety_critical", False))
    safety_penalty = -1.0 if safety_critical and expected_actions and not action_match else 0.0

    base_result["predicted_action"] = predicted_action
    base_result["action_reward"] = float(action_reward)
    base_result["safety_penalty"] = float(safety_penalty)
    base_result["score"] = float(base_result["score"] + action_reward + safety_penalty)
    return base_result
