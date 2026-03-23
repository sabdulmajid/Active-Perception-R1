#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from active_perception_r1.rewards.active_vision_reward import compute_score, extract_final_answer, normalize_answer
from active_perception_r1.sim.live_reinjection import run_live_reinjection_episode
from active_perception_r1.utils.trace_parser import parse_reasoning_trace


@dataclass
class DocVqaExample:
    example_id: str
    image: Image.Image
    question: str
    answer: str
    answer_aliases: list[str]
    bbox_norm: tuple[float, float, float, float] | None


def _normalize_bbox(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    if len(bbox) != 4:
        return None
    x0, y0, x1, y1 = [float(v) for v in bbox]
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 1.5:
        if width <= 0 or height <= 0:
            return None
        x0 /= width
        y0 /= height
        x1 /= width
        y1 /= height
    x0 = max(0.0, min(1.0, x0))
    y0 = max(0.0, min(1.0, y0))
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _bbox_from_answer_words(words: list[str], boxes: list[list[float]], answer: str, image_w: int, image_h: int) -> tuple[float, float, float, float] | None:
    answer_tokens = [token for token in normalize_answer(answer).split() if token]
    if not answer_tokens:
        return None

    hit_boxes: list[tuple[float, float, float, float]] = []
    for word, box in zip(words, boxes):
        if normalize_answer(str(word)) in answer_tokens:
            norm_box = _normalize_bbox(box, image_w, image_h)
            if norm_box is not None:
                hit_boxes.append(norm_box)

    if not hit_boxes:
        return None

    x0 = min(item[0] for item in hit_boxes)
    y0 = min(item[1] for item in hit_boxes)
    x1 = max(item[2] for item in hit_boxes)
    y1 = max(item[3] for item in hit_boxes)

    pad_x = 0.02
    pad_y = 0.02
    x0 = max(0.0, x0 - pad_x)
    y0 = max(0.0, y0 - pad_y)
    x1 = min(1.0, x1 + pad_x)
    y1 = min(1.0, y1 + pad_y)

    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def load_docvqa_examples(sample_count: int, seed: int, split: str = "test") -> list[DocVqaExample]:
    from datasets import load_dataset

    dataset = load_dataset("nielsr/docvqa_1200_examples", split=split)
    shuffled = dataset.shuffle(seed=seed).select(range(min(sample_count, len(dataset))))

    examples: list[DocVqaExample] = []
    for row in shuffled:
        image = row["image"].convert("RGB")
        query = row.get("query")
        if isinstance(query, dict):
            question = str(query.get("en") or query.get("de") or query.get("es") or "").strip()
        else:
            question = str(query or "").strip()

        answers_raw = row.get("answers") or []
        answer_aliases = [str(item).strip() for item in answers_raw if str(item).strip()]

        answer_obj = row.get("answer")
        answer_text = ""
        if isinstance(answer_obj, dict):
            answer_text = str(answer_obj.get("text") or answer_obj.get("matched_text") or "").strip()
        else:
            answer_text = str(answer_obj or "").strip()

        if answer_text:
            answer = answer_text
            if answer_text not in answer_aliases:
                answer_aliases = [answer_text, *answer_aliases]
        else:
            answer = answer_aliases[0] if answer_aliases else ""
        if not question or not answer:
            continue

        words = row.get("words") or []
        boxes = row.get("bounding_boxes") or []
        bbox_norm = _bbox_from_answer_words(words, boxes, answer, image.width, image.height)

        examples.append(
            DocVqaExample(
                example_id=str(row.get("id", len(examples))),
                image=image,
                question=question,
                answer=answer,
                answer_aliases=answer_aliases,
                bbox_norm=bbox_norm,
            )
        )

    return examples


class VisionBenchModel:
    def __init__(self, model_id: str, max_new_tokens: int) -> None:
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as AutoVisionModel
        except ImportError:
            from transformers import AutoModelForCausalLM as AutoVisionModel

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoVisionModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.max_new_tokens = max_new_tokens

    def generate(self, images: list[Image.Image], text: str) -> str:
        message = {
            "role": "user",
            "content": ([{"type": "image"} for _ in images] + [{"type": "text", "text": text}]),
        }
        prompt = self.processor.apply_chat_template([message], add_generation_prompt=True)
        inputs = self.processor(images=images, text=prompt, return_tensors="pt")

        model_device = self.model.device
        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                inputs[key] = value.to(model_device)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )

        input_len = inputs["input_ids"].shape[1]
        completion_ids = generated[:, input_len:]
        text_out = self.processor.batch_decode(completion_ids, skip_special_tokens=True)[0]
        return text_out.strip()


def crop_from_bbox(image: Image.Image, bbox_norm: tuple[float, float, float, float]) -> Image.Image:
    w, h = image.size
    x0 = max(0, min(w, int(round(bbox_norm[0] * w))))
    y0 = max(0, min(h, int(round(bbox_norm[1] * h))))
    x1 = max(x0 + 1, min(w, int(round(bbox_norm[2] * w))))
    y1 = max(y0 + 1, min(h, int(round(bbox_norm[3] * h))))
    return image.crop((x0, y0, x1, y1))


def crop_from_zoom_trace(image: Image.Image, response: str) -> Image.Image | None:
    parsed = parse_reasoning_trace(response)
    for candidate in parsed.zoom_calls:
        if candidate.is_well_formed():
            x0, y0, x1, y1 = candidate.to_normalized_bbox(image.width, image.height)
            return crop_from_bbox(image, (x0, y0, x1, y1))
    return None


def _run_active_default(model: VisionBenchModel, image: Image.Image, question: str) -> tuple[str, str, bool]:
    prompt = (
        f"Question: {question}\n"
        "Think step by step. If visual detail is needed, use <zoom_roi x0=\"...\" y0=\"...\" x1=\"...\" y1=\"...\" />. "
        "Then answer in <answer>...</answer>."
    )
    live_result = run_live_reinjection_episode(
        image=image,
        task_text=prompt,
        generator=model.generate,
        max_steps=3,
    )
    first = live_result.steps[0].response if live_result.steps else ""
    return first, live_result.final_response, live_result.used_zoom_count > 0


def _run_active_strict(model: VisionBenchModel, image: Image.Image, question: str) -> tuple[str, str, bool]:
    zoom_prompt = (
        f"Question: {question}\n"
        "Output ONLY one valid zoom tag and nothing else: "
        "<zoom_roi x0=\"0.00\" y0=\"0.00\" x1=\"1.00\" y1=\"1.00\" />"
    )
    zoom_response = model.generate([image], zoom_prompt)
    crop = crop_from_zoom_trace(image, zoom_response)

    if crop is None:
        retry = "Output ONLY a valid zoom tag in the same format."
        zoom_response = model.generate([image], retry)
        crop = crop_from_zoom_trace(image, zoom_response)

    if crop is None:
        answer_only = model.generate([image], f"Question: {question}\nReturn only <answer>...</answer>.")
        return zoom_response, answer_only, False

    answer = model.generate(
        [image, crop],
        f"Question: {question}\nUse the crop for evidence and answer only in <answer>...</answer>.",
    )
    return zoom_response, answer, True


def run_one_example(model: VisionBenchModel, ex: DocVqaExample, strategy: str) -> dict[str, Any]:
    image = ex.image

    baseline_response = model.generate(
        [image],
        f"Question: {ex.question}\nReturn only the short final answer inside <answer>...</answer>.",
    )

    if ex.bbox_norm is None:
        oracle_response = baseline_response
    else:
        crop = crop_from_bbox(image, ex.bbox_norm)
        oracle_response = model.generate(
            [crop],
            f"Question: {ex.question}\nAnswer from this crop only. Return <answer>...</answer>.",
        )

    if strategy == "strict_zoom":
        active_first, active_final, used_crop = _run_active_strict(model, image, ex.question)
    else:
        active_first, active_final, used_crop = _run_active_default(model, image, ex.question)

    normalized_aliases = {normalize_answer(item) for item in [ex.answer, *ex.answer_aliases] if item}
    gt_norm = normalize_answer(ex.answer)
    baseline_pred = normalize_answer(extract_final_answer(baseline_response))
    oracle_pred = normalize_answer(extract_final_answer(oracle_response))
    active_pred = normalize_answer(extract_final_answer(active_final))

    baseline_acc = float(baseline_pred in normalized_aliases)
    oracle_acc = float(oracle_pred in normalized_aliases)
    active_acc = float(active_pred in normalized_aliases)

    extra_info: dict[str, Any] = {
        "requires_zoom": True,
        "image_size": {"width": image.width, "height": image.height},
        "answer_aliases": [ex.answer, *ex.answer_aliases],
    }
    if ex.bbox_norm is not None:
        extra_info["relevant_regions"] = [{"label": "answer_region", "bbox": list(ex.bbox_norm), "weight": 1.0}]

    reward = compute_score(
        data_source="docvqa_active_perception",
        solution_str=f"{active_first}\n{active_final}",
        ground_truth=ex.answer,
        extra_info=extra_info,
    )

    return {
        "id": ex.example_id,
        "baseline_acc": baseline_acc,
        "oracle_acc": oracle_acc,
        "active_acc": active_acc,
        "active_used_crop": bool(used_crop),
        "has_bbox": bool(ex.bbox_norm is not None),
        "active_reward": float(reward["score"]),
    }


def summarize(rows: list[dict[str, Any]], model_id: str, strategy: str, split: str, sample_count: int, seed: int, elapsed: float) -> dict[str, Any]:
    baseline = statistics.mean(r["baseline_acc"] for r in rows) if rows else 0.0
    oracle = statistics.mean(r["oracle_acc"] for r in rows) if rows else 0.0
    active = statistics.mean(r["active_acc"] for r in rows) if rows else 0.0
    crop_usage = statistics.mean(float(r["active_used_crop"]) for r in rows) if rows else 0.0
    bbox_coverage = statistics.mean(float(r["has_bbox"]) for r in rows) if rows else 0.0
    reward_mean = statistics.mean(r["active_reward"] for r in rows) if rows else 0.0

    return {
        "model_id": model_id,
        "strategy": strategy,
        "split": split,
        "samples": sample_count,
        "seed": seed,
        "duration_sec": elapsed,
        "baseline_acc": baseline,
        "oracle_acc": oracle,
        "active_acc": active,
        "active_crop_usage": crop_usage,
        "bbox_coverage": bbox_coverage,
        "active_reward_mean": reward_mean,
        "active_minus_baseline": active - baseline,
        "oracle_minus_baseline": oracle - baseline,
    }


def render_markdown(run_summaries: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        "# Real-Data VLM Benchmark Suite",
        "",
        "Benchmark on `nielsr/docvqa_1200_examples` with active-perception strategies.",
        "",
        "| Model | Strategy | Samples | Baseline | Active | Active-Baseline | Oracle | Crop Usage |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in run_summaries:
        lines.append(
            "| {model} | {strategy} | {samples} | {baseline:.4f} | {active:.4f} | {delta:.4f} | {oracle:.4f} | {crop:.4f} |".format(
                model=row["model_id"],
                strategy=row["strategy"],
                samples=row["samples"],
                baseline=row["baseline_acc"],
                active=row["active_acc"],
                delta=row["active_minus_baseline"],
                oracle=row["oracle_acc"],
                crop=row["active_crop_usage"],
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-model active-perception benchmark on DocVQA examples.")
    parser.add_argument("--models", required=True, help="Comma-separated model ids")
    parser.add_argument("--strategies", default="default,strict_zoom", help="Comma-separated strategies")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output-dir", default="reports/real_benchmark")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_docvqa_examples(sample_count=args.samples, seed=args.seed, split=args.split)

    model_ids = [item.strip() for item in args.models.split(",") if item.strip()]
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]

    run_summaries: list[dict[str, Any]] = []

    for model_id in model_ids:
        try:
            model = VisionBenchModel(model_id=model_id, max_new_tokens=args.max_new_tokens)
        except Exception as error:
            run_summaries.append(
                {
                    "model_id": model_id,
                    "strategy": "load_error",
                    "split": args.split,
                    "samples": len(examples),
                    "seed": args.seed,
                    "duration_sec": 0.0,
                    "baseline_acc": 0.0,
                    "oracle_acc": 0.0,
                    "active_acc": 0.0,
                    "active_crop_usage": 0.0,
                    "bbox_coverage": 0.0,
                    "active_reward_mean": 0.0,
                    "active_minus_baseline": 0.0,
                    "oracle_minus_baseline": 0.0,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue

        for strategy in strategies:
            started = time.time()
            rows: list[dict[str, Any]] = []

            for ex in examples:
                try:
                    rows.append(run_one_example(model, ex, strategy))
                except RuntimeError as error:
                    if "out of memory" in str(error).lower():
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise

            elapsed = time.time() - started
            run_summaries.append(
                summarize(
                    rows=rows,
                    model_id=model_id,
                    strategy=strategy,
                    split=args.split,
                    sample_count=len(rows),
                    seed=args.seed,
                    elapsed=elapsed,
                )
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    json_path = output_dir / f"docvqa-suite-{timestamp}.json"
    md_path = output_dir / f"docvqa-suite-{timestamp}.md"

    json_path.write_text(json.dumps(run_summaries, indent=2), encoding="utf-8")
    render_markdown(run_summaries, md_path)

    print(f"Wrote JSON summary: {json_path}")
    print(f"Wrote Markdown summary: {md_path}")
    for item in run_summaries:
        if item.get("strategy") == "load_error":
            print(f"LOAD_ERROR {item['model_id']}: {item.get('error')}")
        else:
            print(
                f"{item['model_id']} [{item['strategy']}] samples={item['samples']} "
                f"baseline={item['baseline_acc']:.4f} active={item['active_acc']:.4f} "
                f"delta={item['active_minus_baseline']:.4f} crop={item['active_crop_usage']:.4f}"
            )


if __name__ == "__main__":
    main()
