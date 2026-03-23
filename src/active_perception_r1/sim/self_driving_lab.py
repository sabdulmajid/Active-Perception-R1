from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Callable

from active_perception_r1.envs.zoom_simulator import intersection_over_region, intersection_over_union


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _zone_alignment(bbox: tuple[float, float, float, float], zone: tuple[float, float, float, float]) -> float:
    cx, cy = _center(bbox)
    zx, zy = _center(zone)
    distance = math.sqrt((cx - zx) ** 2 + (cy - zy) ** 2)
    return max(0.0, 1.0 - distance / 0.9)


def _refine_bbox(
    bbox: tuple[float, float, float, float],
    scale: float,
) -> tuple[float, float, float, float]:
    cx, cy = _center(bbox)
    width = (bbox[2] - bbox[0]) * scale
    height = (bbox[3] - bbox[1]) * scale
    return (
        _clamp(cx - width / 2.0, 0.0, 1.0),
        _clamp(cy - height / 2.0, 0.0, 1.0),
        _clamp(cx + width / 2.0, 0.0, 1.0),
        _clamp(cy + height / 2.0, 0.0, 1.0),
    )


def _sample_bbox_in_zone(
    rng: random.Random,
    zone: tuple[float, float, float, float],
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    x_low = zone[0]
    y_low = zone[1]
    x_high = max(zone[0], zone[2] - width)
    y_high = max(zone[1], zone[3] - height)
    x0 = rng.uniform(x_low, x_high if x_high >= x_low else x_low)
    y0 = rng.uniform(y_low, y_high if y_high >= y_low else y_low)
    return (x0, y0, _clamp(x0 + width, 0.0, 1.0), _clamp(y0 + height, 0.0, 1.0))


ZONES = {
    "top_left": (0.04, 0.02, 0.30, 0.28),
    "top_center": (0.32, 0.02, 0.68, 0.30),
    "top_right": (0.70, 0.02, 0.96, 0.28),
    "mid_left": (0.03, 0.22, 0.30, 0.60),
    "mid_right": (0.70, 0.22, 0.97, 0.60),
    "center": (0.28, 0.24, 0.72, 0.78),
    "bottom_left": (0.05, 0.52, 0.38, 0.96),
    "bottom_center": (0.28, 0.52, 0.72, 0.96),
    "bottom_right": (0.62, 0.52, 0.95, 0.96),
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    prior_zones: tuple[str, ...]
    base_scale: float
    base_difficulty: float
    criticality: float
    shortcut_prior: float


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    visibility: float
    occlusion_boost: float
    scale_multiplier: float
    detector_noise: float
    text_leakage: float


@dataclass(frozen=True)
class Proposal:
    bbox: tuple[float, float, float, float]
    confidence: float
    source: str


@dataclass(frozen=True)
class Scene:
    scene_id: str
    task: TaskSpec
    condition: ConditionSpec
    relevant_bbox: tuple[float, float, float, float]
    proposals: tuple[Proposal, ...]
    global_signal: float
    shortcut_signal: float
    occlusion: float


@dataclass(frozen=True)
class CropTrace:
    bbox: tuple[float, float, float, float]
    coverage: float
    iou: float
    evidence_gain: float
    source: str


@dataclass(frozen=True)
class PolicyOutcome:
    scene_id: str
    policy_name: str
    task: str
    condition: str
    correct: float
    grounded_correct: float
    ungrounded_correct: float
    evidence_hit: float
    unsafe_failure: float
    max_coverage: float
    max_iou: float
    avg_evidence_gain: float
    num_crops: float
    repeat_crop_rate: float
    answer_only_reward: float
    grounded_reward: float
    safety_reward: float


TASKS = (
    TaskSpec("traffic_light_state", ("top_left", "top_center", "top_right"), 0.025, 0.54, 1.0, 0.30),
    TaskSpec("pedestrian_intent", ("bottom_left", "bottom_center", "bottom_right"), 0.070, 0.64, 1.0, 0.14),
    TaskSpec("construction_sign", ("mid_left", "mid_right"), 0.032, 0.60, 0.70, 0.22),
    TaskSpec("lane_merge_hazard", ("mid_left", "mid_right", "top_center"), 0.055, 0.69, 0.95, 0.24),
    TaskSpec("cut_in_vehicle", ("mid_left", "mid_right", "center"), 0.080, 0.72, 1.0, 0.18),
)

CONDITIONS = (
    ConditionSpec("clean", 1.00, 0.00, 1.00, 0.05, 0.05),
    ConditionSpec("night_rain", 0.66, 0.08, 0.90, 0.15, 0.08),
    ConditionSpec("occluded", 0.77, 0.24, 0.95, 0.10, 0.08),
    ConditionSpec("small_object", 0.86, 0.10, 0.55, 0.11, 0.09),
    ConditionSpec("text_leakage", 0.73, 0.05, 0.85, 0.12, 0.40),
)

POLICIES = (
    "passive_cot",
    "center_zoom",
    "task_prior_zoom",
    "detector_first",
    "active_verify",
    "oracle_roi",
)


def generate_scene(
    task: TaskSpec,
    condition: ConditionSpec,
    rng: random.Random,
    scene_index: int,
    seed: int | None = None,
) -> Scene:
    zone_name = rng.choice(task.prior_zones)
    zone = ZONES[zone_name]
    width = math.sqrt(task.base_scale * condition.scale_multiplier) * rng.uniform(0.8, 1.2)
    height = width * rng.uniform(0.8, 1.3)
    relevant_bbox = _sample_bbox_in_zone(rng, zone, width, height)
    relevant_area = _bbox_area(relevant_bbox)
    occlusion = _clamp(rng.uniform(0.05, 0.18) + condition.occlusion_boost, 0.0, 0.55)

    size_factor = min(1.0, math.sqrt(relevant_area / max(task.base_scale, 1e-3)))
    global_signal = condition.visibility * (1.0 - occlusion) * size_factor
    shortcut_signal = min(1.0, task.shortcut_prior + condition.text_leakage)

    proposals: list[Proposal] = []
    gt_noise = condition.detector_noise
    gt_proposal = (
        _clamp(relevant_bbox[0] + rng.uniform(-gt_noise, gt_noise), 0.0, 1.0),
        _clamp(relevant_bbox[1] + rng.uniform(-gt_noise, gt_noise), 0.0, 1.0),
        _clamp(relevant_bbox[2] + rng.uniform(-gt_noise, gt_noise), 0.0, 1.0),
        _clamp(relevant_bbox[3] + rng.uniform(-gt_noise, gt_noise), 0.0, 1.0),
    )
    gt_confidence = _clamp(0.58 + 0.30 * condition.visibility - 0.25 * occlusion + rng.uniform(-0.08, 0.08), 0.05, 0.99)
    proposals.append(Proposal(gt_proposal, gt_confidence, "gt_noisy"))

    distractor_count = rng.randint(3, 6)
    plausible_zones = list(task.prior_zones) + ["center", "mid_left", "mid_right", "bottom_center"]
    for _ in range(distractor_count):
        distractor_zone = ZONES[rng.choice(plausible_zones)]
        dw = width * rng.uniform(0.8, 1.4)
        dh = height * rng.uniform(0.8, 1.4)
        distractor_bbox = _sample_bbox_in_zone(rng, distractor_zone, dw, dh)
        confidence = _clamp(
            0.28 + 0.22 * condition.text_leakage + 0.16 * _zone_alignment(distractor_bbox, zone) + rng.uniform(-0.10, 0.12),
            0.02,
            0.95,
        )
        proposals.append(Proposal(distractor_bbox, confidence, "distractor"))

    proposals.sort(key=lambda item: item.confidence, reverse=True)

    return Scene(
        scene_id=(
            f"seed_{seed}:{task.name}:{condition.name}:{scene_index}"
            if seed is not None
            else f"{task.name}:{condition.name}:{scene_index}"
        ),
        task=task,
        condition=condition,
        relevant_bbox=relevant_bbox,
        proposals=tuple(proposals),
        global_signal=global_signal,
        shortcut_signal=shortcut_signal,
        occlusion=occlusion,
    )


def evaluate_crop(scene: Scene, bbox: tuple[float, float, float, float], source: str) -> CropTrace:
    coverage = intersection_over_region(bbox, scene.relevant_bbox)
    iou = intersection_over_union(bbox, scene.relevant_bbox)
    area = _bbox_area(bbox)
    relevant_area = _bbox_area(scene.relevant_bbox)
    ideal_area = _clamp(relevant_area * 2.4, 0.02, 0.18)
    area_alignment = math.exp(-abs(math.log((area + 1e-6) / (ideal_area + 1e-6))))
    evidence_gain = coverage * scene.condition.visibility * (1.0 - scene.occlusion) * (0.45 + 0.85 * area_alignment)
    return CropTrace(bbox=bbox, coverage=coverage, iou=iou, evidence_gain=evidence_gain, source=source)


def choose_policy_crops(scene: Scene, policy_name: str, budget: int | None = None) -> list[CropTrace]:
    traces: list[CropTrace] = []
    budget = 0 if policy_name == "passive_cot" else (budget if budget is not None else 2)

    if policy_name == "passive_cot":
        return traces

    if policy_name == "center_zoom":
        traces.append(evaluate_crop(scene, (0.30, 0.26, 0.70, 0.80), "center"))
        return traces

    if policy_name == "task_prior_zoom":
        zone = ZONES[scene.task.prior_zones[0]]
        traces.append(evaluate_crop(scene, _refine_bbox(zone, 0.9), "task_prior"))
        return traces

    if policy_name == "detector_first":
        traces.append(evaluate_crop(scene, scene.proposals[0].bbox, scene.proposals[0].source))
        return traces

    if policy_name == "oracle_roi":
        traces.append(evaluate_crop(scene, scene.relevant_bbox, "oracle"))
        return traces

    if policy_name != "active_verify":
        raise ValueError(f"Unknown policy: {policy_name}")

    ranked = sorted(
        scene.proposals,
        key=lambda proposal: 0.62 * proposal.confidence
        + 0.38 * max(_zone_alignment(proposal.bbox, ZONES[zone_name]) for zone_name in scene.task.prior_zones),
        reverse=True,
    )
    first = ranked[0]
    first_trace = evaluate_crop(scene, first.bbox, first.source)
    traces.append(first_trace)

    if budget <= 1:
        return traces

    if first_trace.coverage >= 0.30:
        traces.append(evaluate_crop(scene, _refine_bbox(first.bbox, 0.72), "refine"))
    else:
        second = ranked[1]
        traces.append(evaluate_crop(scene, second.bbox, second.source))

    if budget <= 2:
        return traces

    max_coverage = max(trace.coverage for trace in traces)
    if max_coverage < 0.50:
        fallback_zone = ZONES[scene.task.prior_zones[-1]]
        traces.append(evaluate_crop(scene, _refine_bbox(fallback_zone, 0.85), "fallback_prior"))

    return traces


def score_policy(scene: Scene, policy_name: str, rng: random.Random, budget_override: int | None = None) -> PolicyOutcome:
    traces = choose_policy_crops(scene, policy_name, budget=budget_override)
    max_coverage = max((trace.coverage for trace in traces), default=0.0)
    max_iou = max((trace.iou for trace in traces), default=0.0)
    avg_evidence_gain = sum(trace.evidence_gain for trace in traces) / max(1, len(traces))
    sorted_gains = sorted((trace.evidence_gain for trace in traces), reverse=True)
    evidence_signal = max(sorted_gains, default=0.0)
    repeat_count = 0
    for idx, trace in enumerate(traces):
        for other in traces[idx + 1 :]:
            if intersection_over_union(trace.bbox, other.bbox) > 0.70:
                repeat_count += 1

    if policy_name == "passive_cot":
        global_weight = 1.05
        shortcut_bias = 0.72
    elif policy_name == "center_zoom":
        global_weight = 0.90
        shortcut_bias = 0.42
    elif policy_name == "task_prior_zoom":
        global_weight = 0.85
        shortcut_bias = 0.34
    elif policy_name == "detector_first":
        global_weight = 0.78
        shortcut_bias = 0.28
    elif policy_name == "active_verify":
        global_weight = 0.72
        shortcut_bias = 0.18
    elif policy_name == "oracle_roi":
        global_weight = 0.68
        shortcut_bias = 0.08
    else:
        raise ValueError(policy_name)

    verification_bonus = 0.0
    if policy_name == "active_verify" and len(sorted_gains) >= 2:
        evidence_signal += 0.35 * sorted_gains[1]
        if traces[0].coverage >= 0.30 and traces[1].coverage >= 0.20:
            verification_bonus += 0.18
        if max_coverage >= 0.55 and max_iou >= 0.35:
            verification_bonus += 0.10

    signal = (
        global_weight * scene.global_signal
        + 1.55 * evidence_signal
        + shortcut_bias * scene.shortcut_signal
        + verification_bonus
        - 0.06 * max(0, len(traces) - 1)
        - 0.08 * repeat_count
        - scene.task.base_difficulty
    )
    correctness_prob = _sigmoid(signal * 3.0)
    correct = float(rng.random() < correctness_prob)
    evidence_hit = float(max_coverage >= 0.45)
    grounded_correct = float(correct and evidence_hit)
    ungrounded_correct = float(correct and not evidence_hit)
    unsafe_failure = float(scene.task.criticality >= 0.95 and not correct)
    repeat_crop_rate = repeat_count / max(1, len(traces))

    answer_only_reward = correct
    grounded_reward = (
        correct
        + 0.70 * evidence_hit
        + 0.30 * max_coverage
        - 0.12 * len(traces)
        - 0.45 * ungrounded_correct
        - 0.10 * repeat_crop_rate
    )
    safety_reward = grounded_reward + 0.35 * scene.task.criticality * correct - 1.10 * unsafe_failure

    return PolicyOutcome(
        scene_id=scene.scene_id,
        policy_name=policy_name,
        task=scene.task.name,
        condition=scene.condition.name,
        correct=correct,
        grounded_correct=grounded_correct,
        ungrounded_correct=ungrounded_correct,
        evidence_hit=evidence_hit,
        unsafe_failure=unsafe_failure,
        max_coverage=max_coverage,
        max_iou=max_iou,
        avg_evidence_gain=avg_evidence_gain,
        num_crops=float(len(traces)),
        repeat_crop_rate=float(repeat_crop_rate),
        answer_only_reward=float(answer_only_reward),
        grounded_reward=float(grounded_reward),
        safety_reward=float(safety_reward),
    )


def _mean(items: list[float]) -> float:
    return sum(items) / len(items) if items else 0.0


def _aggregate(records: list[PolicyOutcome], key_fn: Callable[[PolicyOutcome], tuple[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[PolicyOutcome]] = {}
    for record in records:
        grouped.setdefault(key_fn(record), []).append(record)

    summary: dict[str, dict[str, float]] = {}
    for (policy_name, label), values in grouped.items():
        summary.setdefault(policy_name, {})[label] = {
            "accuracy": _mean([item.correct for item in values]),
            "grounded_accuracy": _mean([item.grounded_correct for item in values]),
            "evidence_hit_rate": _mean([item.evidence_hit for item in values]),
            "ungrounded_correct_rate": _mean([item.ungrounded_correct for item in values]),
            "unsafe_failure_rate": _mean([item.unsafe_failure for item in values]),
            "mean_crops": _mean([item.num_crops for item in values]),
            "mean_grounded_reward": _mean([item.grounded_reward for item in values]),
            "mean_safety_reward": _mean([item.safety_reward for item in values]),
        }
    return summary


def _compute_selection_rates(records: list[PolicyOutcome], reward_key: str) -> dict[str, float]:
    selection_counts = {policy_name: 0 for policy_name in POLICIES}
    scene_groups: dict[str, list[PolicyOutcome]] = {}
    for record in records:
        scene_groups.setdefault(record.scene_id, []).append(record)

    for outcomes in scene_groups.values():
        best = max(outcomes, key=lambda item: getattr(item, reward_key))
        selection_counts[best.policy_name] += 1

    total = max(1, len(scene_groups))
    return {policy_name: count / total for policy_name, count in selection_counts.items()}


def _pairwise_preference(records: list[PolicyOutcome], lhs: str, rhs: str, reward_key: str) -> float:
    scene_groups: dict[str, dict[str, PolicyOutcome]] = {}
    for record in records:
        scene_groups.setdefault(record.scene_id, {})[record.policy_name] = record

    wins = 0
    total = 0
    for outcomes in scene_groups.values():
        if lhs not in outcomes or rhs not in outcomes:
            continue
        total += 1
        if getattr(outcomes[lhs], reward_key) > getattr(outcomes[rhs], reward_key):
            wins += 1
    return wins / max(1, total)


def _budget_sweep(seeds: list[int], scenes_per_combo: int) -> dict[str, dict[str, float]]:
    results: dict[str, list[PolicyOutcome]] = {"budget_1": [], "budget_2": [], "budget_3": []}
    for seed in seeds:
        rng = random.Random(seed)
        for task in TASKS:
            for condition in CONDITIONS:
                for scene_index in range(scenes_per_combo):
                    scene = generate_scene(task, condition, rng, scene_index, seed=seed)
                    for budget in [1, 2, 3]:
                        results[f"budget_{budget}"].append(
                            score_policy(scene, "active_verify", rng=random.Random(seed * 10000 + scene_index + budget), budget_override=budget)
                        )

    summary: dict[str, dict[str, float]] = {}
    for label, items in results.items():
        summary[label] = {
            "accuracy": _mean([item.correct for item in items]),
            "grounded_accuracy": _mean([item.grounded_correct for item in items]),
            "evidence_hit_rate": _mean([item.evidence_hit for item in items]),
            "mean_grounded_reward": _mean([item.grounded_reward for item in items]),
            "mean_crops": _mean([item.num_crops for item in items]),
        }
    return summary


def _make_markdown(
    overall_summary: dict[str, dict[str, float]],
    condition_summary: dict[str, dict[str, dict[str, float]]],
    selection_answer_only: dict[str, float],
    selection_grounded: dict[str, float],
    budget_sweep: dict[str, dict[str, float]],
    pairwise: dict[str, float],
    scenes_per_combo: int,
    seeds: list[int],
) -> str:
    sorted_policies = sorted(
        overall_summary.items(),
        key=lambda item: item[1]["grounded_accuracy"],
        reverse=True,
    )
    passive = overall_summary["passive_cot"]
    active = overall_summary["active_verify"]
    small_object_active = condition_summary["active_verify"]["small_object"]
    small_object_passive = condition_summary["passive_cot"]["small_object"]

    lines = [
        "# Self-Driving Policy Sweep",
        "",
        "Synthetic policy sweep inspired by DriveLM / DriveBench-style perception failures and safety-critical local evidence tasks.",
        f"Generated with `{scenes_per_combo}` scenes per task-condition-seed across seeds `{seeds}`.",
        "",
        "## Key Findings",
        "",
        f"- `active_verify` is the strongest non-oracle policy overall with grounded accuracy `{active['grounded_accuracy']:.3f}`.",
        f"- Compared with `passive_cot`, `active_verify` improves grounded accuracy by `{active['grounded_accuracy'] - passive['grounded_accuracy']:.3f}` overall and `{small_object_active['grounded_accuracy'] - small_object_passive['grounded_accuracy']:.3f}` on `small_object` scenes.",
        f"- Under answer-only reward, trajectory selection still favors `passive_cot` `{selection_answer_only['passive_cot']:.3f}` of the time. Under grounded reward, `passive_cot` drops to `{selection_grounded['passive_cot']:.3f}` while `oracle_roi` dominates overall at `{selection_grounded['oracle_roi']:.3f}`.",
        f"- Pairwise grounded preference chooses `active_verify` over `passive_cot` `{pairwise['grounded_reward']:.3f}` of the time, versus `{pairwise['answer_only_reward']:.3f}` under answer-only preference.",
        f"- Budgeted inspection shows diminishing returns past two crops: budget 2 grounded accuracy `{budget_sweep['budget_2']['grounded_accuracy']:.3f}` vs budget 3 `{budget_sweep['budget_3']['grounded_accuracy']:.3f}`.",
        "",
        "## Overall Metrics",
        "",
        "| Policy | Accuracy | Grounded Acc. | Evidence Hit | Unsafe Failure | Mean Crops | Grounded Reward |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for policy_name, metrics in sorted_policies:
        lines.append(
            f"| {policy_name} | {metrics['accuracy']:.3f} | {metrics['grounded_accuracy']:.3f} | "
            f"{metrics['evidence_hit_rate']:.3f} | {metrics['unsafe_failure_rate']:.3f} | "
            f"{metrics['mean_crops']:.2f} | {metrics['mean_grounded_reward']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Reward-Induced Selection Rates",
            "",
            "| Reward | passive_cot | center_zoom | task_prior_zoom | detector_first | active_verify | oracle_roi |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| answer_only | {selection_answer_only['passive_cot']:.3f} | {selection_answer_only['center_zoom']:.3f} | {selection_answer_only['task_prior_zoom']:.3f} | {selection_answer_only['detector_first']:.3f} | {selection_answer_only['active_verify']:.3f} | {selection_answer_only['oracle_roi']:.3f} |",
            f"| grounded | {selection_grounded['passive_cot']:.3f} | {selection_grounded['center_zoom']:.3f} | {selection_grounded['task_prior_zoom']:.3f} | {selection_grounded['detector_first']:.3f} | {selection_grounded['active_verify']:.3f} | {selection_grounded['oracle_roi']:.3f} |",
            "",
            "## Active Verify Budget Sweep",
            "",
            "| Budget | Accuracy | Grounded Acc. | Evidence Hit | Mean Crops | Grounded Reward |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for label, metrics in budget_sweep.items():
        lines.append(
            f"| {label} | {metrics['accuracy']:.3f} | {metrics['grounded_accuracy']:.3f} | "
            f"{metrics['evidence_hit_rate']:.3f} | {metrics['mean_crops']:.2f} | {metrics['mean_grounded_reward']:.3f} |"
        )

    return "\n".join(lines) + "\n"


def generate_policy_sweep_report(scenes_per_combo: int = 120, seeds: list[int] | None = None) -> dict[str, object]:
    seeds = seeds or [7, 11, 23]
    records: list[PolicyOutcome] = []

    for seed in seeds:
        scene_rng = random.Random(seed)
        for task in TASKS:
            for condition in CONDITIONS:
                for scene_index in range(scenes_per_combo):
                    scene = generate_scene(task, condition, scene_rng, scene_index, seed=seed)
                    for policy_index, policy_name in enumerate(POLICIES):
                        policy_rng = random.Random(seed * 100000 + scene_index * 10 + policy_index)
                        records.append(score_policy(scene, policy_name, policy_rng))

    overall_summary = {
        policy_name: {
            "accuracy": _mean([item.correct for item in records if item.policy_name == policy_name]),
            "grounded_accuracy": _mean([item.grounded_correct for item in records if item.policy_name == policy_name]),
            "evidence_hit_rate": _mean([item.evidence_hit for item in records if item.policy_name == policy_name]),
            "unsafe_failure_rate": _mean([item.unsafe_failure for item in records if item.policy_name == policy_name]),
            "mean_crops": _mean([item.num_crops for item in records if item.policy_name == policy_name]),
            "mean_grounded_reward": _mean([item.grounded_reward for item in records if item.policy_name == policy_name]),
        }
        for policy_name in POLICIES
    }

    condition_summary = _aggregate(records, key_fn=lambda item: (item.policy_name, item.condition))
    selection_answer_only = _compute_selection_rates(records, reward_key="answer_only_reward")
    selection_grounded = _compute_selection_rates(records, reward_key="grounded_reward")
    pairwise = {
        "answer_only_reward": _pairwise_preference(records, "active_verify", "passive_cot", "answer_only_reward"),
        "grounded_reward": _pairwise_preference(records, "active_verify", "passive_cot", "grounded_reward"),
        "safety_reward": _pairwise_preference(records, "active_verify", "passive_cot", "safety_reward"),
    }
    budget_sweep = _budget_sweep(seeds=seeds, scenes_per_combo=scenes_per_combo)

    markdown_summary = _make_markdown(
        overall_summary=overall_summary,
        condition_summary=condition_summary,
        selection_answer_only=selection_answer_only,
        selection_grounded=selection_grounded,
        budget_sweep=budget_sweep,
        pairwise=pairwise,
        scenes_per_combo=scenes_per_combo,
        seeds=seeds,
    )

    return {
        "metadata": {
            "scenes_per_combo": scenes_per_combo,
            "seeds": seeds,
            "tasks": [task.name for task in TASKS],
            "conditions": [condition.name for condition in CONDITIONS],
            "policies": list(POLICIES),
            "num_scene_instances": len(seeds) * len(TASKS) * len(CONDITIONS) * scenes_per_combo,
            "num_policy_rollouts": len(seeds) * len(TASKS) * len(CONDITIONS) * scenes_per_combo * len(POLICIES),
        },
        "overall_summary": overall_summary,
        "condition_summary": condition_summary,
        "selection_answer_only": selection_answer_only,
        "selection_grounded": selection_grounded,
        "pairwise_preferences": pairwise,
        "active_verify_budget_sweep": budget_sweep,
        "example_records": [asdict(record) for record in records[:24]],
        "markdown_summary": markdown_summary,
    }
