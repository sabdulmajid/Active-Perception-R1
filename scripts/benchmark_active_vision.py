#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from active_perception_r1.rewards.active_vision_reward import (
    compute_score,
    extract_final_answer,
    normalize_answer,
)
from active_perception_r1.utils.trace_parser import parse_reasoning_trace


@dataclass
class BenchmarkExample:
    id: str
    image_path: Path
    ground_truth: str
    bbox_norm: tuple[float, float, float, float]
    extra_info: dict[str, Any]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalized_to_pixels(
    bbox_norm: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox_norm
    return (
        int(round(clamp(x0, 0.0, 1.0) * width)),
        int(round(clamp(y0, 0.0, 1.0) * height)),
        int(round(clamp(x1, 0.0, 1.0) * width)),
        int(round(clamp(y1, 0.0, 1.0) * height)),
    )


def parse_int_answer(text: str) -> str:
    candidate = extract_final_answer(text)
    candidate = "".join(ch for ch in candidate if ch.isdigit())
    if candidate:
        return candidate
    return normalize_answer(extract_final_answer(text))


def make_synthetic_dataset(
    output_dir: Path,
    sample_count: int,
    seed: int,
    width: int = 1024,
    height: int = 768,
) -> list[BenchmarkExample]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    examples: list[BenchmarkExample] = []

    for idx in range(sample_count):
        value = rng.randint(10, 99)

        image = Image.new("RGB", (width, height), color=(248, 248, 248))
        draw = ImageDraw.Draw(image)

        for _ in range(40):
            x0 = rng.randint(0, width - 100)
            y0 = rng.randint(0, height - 100)
            x1 = x0 + rng.randint(20, 220)
            y1 = y0 + rng.randint(12, 120)
            color = (rng.randint(150, 220), rng.randint(150, 220), rng.randint(150, 220))
            draw.rectangle((x0, y0, x1, y1), outline=color, width=1)

        box_w = int(width * 0.23)
        box_h = int(height * 0.16)
        x0 = int(width * rng.uniform(0.65, 0.74))
        y0 = int(height * rng.uniform(0.04, 0.12))
        x1 = x0 + box_w
        y1 = y0 + box_h

        draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill=(250, 250, 220), outline=(10, 10, 10), width=3)
        draw.text((x0 + 14, y0 + 10), "INSET", fill=(0, 0, 0))
        draw.text((x0 + 14, y0 + 42), f"BLUE={value}", fill=(0, 40, 180))

        image_path = output_dir / f"sample_{idx:03d}.png"
        image.save(image_path)

        bbox_norm = (x0 / width, y0 / height, x1 / width, y1 / height)
        extra_info = {
            "requires_zoom": True,
            "image_size": {"width": width, "height": height},
            "relevant_regions": [{"label": "top_right_inset", "bbox": list(bbox_norm), "weight": 1.0}],
            "answer_aliases": [str(value), f"{value}.0"],
        }
        examples.append(
            BenchmarkExample(
                id=f"sample_{idx:03d}",
                image_path=image_path,
                ground_truth=str(value),
                bbox_norm=bbox_norm,
                extra_info=extra_info,
            )
        )

    return examples


class VisionBenchModel:
    def __init__(self, model_id: str, max_new_tokens: int) -> None:
        from transformers import AutoModelForVision2Seq, AutoProcessor

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
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
    x0, y0, x1, y1 = normalized_to_pixels(bbox_norm, image.width, image.height)
    if x1 <= x0:
        x1 = min(image.width, x0 + 1)
    if y1 <= y0:
        y1 = min(image.height, y0 + 1)
    return image.crop((x0, y0, x1, y1))


def crop_from_zoom_trace(image: Image.Image, response: str) -> Image.Image | None:
    parsed = parse_reasoning_trace(response)
    if not parsed.zoom_calls:
        return None
    first = parsed.zoom_calls[0]
    if not first.is_well_formed():
        return None
    bbox_norm = first.to_normalized_bbox(image.width, image.height)
    return crop_from_bbox(image, bbox_norm)


def run_example(
    model: VisionBenchModel,
    example: BenchmarkExample,
) -> dict[str, Any]:
    image = Image.open(example.image_path).convert("RGB")

    baseline_prompt = (
        "Read the blue numeric value in the top-right inset and answer with only digits inside "
        "<answer>...</answer>."
    )
    baseline_response = model.generate([image], baseline_prompt)

    oracle_crop = crop_from_bbox(image, example.bbox_norm)
    oracle_prompt = "Read the numeric value in this crop and answer only as <answer>...</answer>."
    oracle_response = model.generate([oracle_crop], oracle_prompt)

    active_prompt_1 = (
        "Think step by step. If you need to inspect details, output one tool tag in your think block as "
        "<zoom_roi x0=\"...\" y0=\"...\" x1=\"...\" y1=\"...\" /> using normalized coordinates, then answer "
        "as <answer>...</answer>."
    )
    active_pass_1 = model.generate([image], active_prompt_1)
    dynamic_crop = crop_from_zoom_trace(image, active_pass_1)

    if dynamic_crop is None:
        active_final_response = active_pass_1
        used_crop = False
    else:
        active_prompt_2 = (
            "You now have the original image and a zoomed crop. Use the crop to verify the value. "
            "Answer only as <answer>...</answer>."
        )
        active_final_response = model.generate([image, dynamic_crop], active_prompt_2)
        used_crop = True

    gt_norm = normalize_answer(example.ground_truth)
    baseline_pred = parse_int_answer(baseline_response)
    oracle_pred = parse_int_answer(oracle_response)
    active_pred = parse_int_answer(active_final_response)

    baseline_acc = float(normalize_answer(str(baseline_pred)) == gt_norm)
    oracle_acc = float(normalize_answer(str(oracle_pred)) == gt_norm)
    active_acc = float(normalize_answer(str(active_pred)) == gt_norm)

    reward_dict = compute_score(
        data_source="active_perception_v0",
        solution_str=active_pass_1 + "\n" + active_final_response,
        ground_truth=example.ground_truth,
        extra_info=example.extra_info,
    )

    return {
        "id": example.id,
        "ground_truth": example.ground_truth,
        "baseline_response": baseline_response,
        "oracle_response": oracle_response,
        "active_pass_1": active_pass_1,
        "active_final_response": active_final_response,
        "baseline_pred": str(baseline_pred),
        "oracle_pred": str(oracle_pred),
        "active_pred": str(active_pred),
        "baseline_acc": baseline_acc,
        "oracle_acc": oracle_acc,
        "active_acc": active_acc,
        "active_used_crop": used_crop,
        "reward_score": reward_dict["score"],
        "reward_visual": reward_dict["visual_perception_reward"],
        "reward_relevant_zoom_count": reward_dict["relevant_zoom_count"],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_acc = statistics.mean(row["baseline_acc"] for row in rows)
    oracle_acc = statistics.mean(row["oracle_acc"] for row in rows)
    active_acc = statistics.mean(row["active_acc"] for row in rows)
    used_crop_rate = statistics.mean(float(row["active_used_crop"]) for row in rows)
    avg_reward = statistics.mean(float(row["reward_score"]) for row in rows)

    return {
        "n": len(rows),
        "baseline_acc": baseline_acc,
        "oracle_crop_acc": oracle_acc,
        "active_two_pass_acc": active_acc,
        "active_crop_usage_rate": used_crop_rate,
        "average_active_reward_score": avg_reward,
        "active_minus_baseline": active_acc - baseline_acc,
        "oracle_minus_baseline": oracle_acc - baseline_acc,
    }


def write_markdown_report(path: Path, model_id: str, summary: dict[str, Any], duration_sec: float) -> None:
    lines = [
        "# Active Perception Benchmark Report",
        "",
        f"- model: `{model_id}`",
        f"- timestamp_utc: `{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}`",
        f"- duration_sec: `{duration_sec:.2f}`",
        f"- samples: `{summary['n']}`",
        "",
        "## Results",
        "",
        f"- baseline_acc: `{summary['baseline_acc']:.4f}`",
        f"- oracle_crop_acc: `{summary['oracle_crop_acc']:.4f}`",
        f"- active_two_pass_acc: `{summary['active_two_pass_acc']:.4f}`",
        f"- active_crop_usage_rate: `{summary['active_crop_usage_rate']:.4f}`",
        f"- avg_active_reward_score: `{summary['average_active_reward_score']:.4f}`",
        f"- delta_active_vs_baseline: `{summary['active_minus_baseline']:.4f}`",
        f"- delta_oracle_vs_baseline: `{summary['oracle_minus_baseline']:.4f}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run active-perception benchmark on a real VLM.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output-dir", default="reports/active_benchmark")
    parser.add_argument("--dataset-dir", default="data/benchmark_synth")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(args.dataset_dir)
    dataset = make_synthetic_dataset(dataset_dir, sample_count=args.samples, seed=args.seed)

    started = time.time()
    model = VisionBenchModel(model_id=args.model_id, max_new_tokens=args.max_new_tokens)

    rows: list[dict[str, Any]] = []
    for sample in dataset:
        rows.append(run_example(model, sample))

    summary = summarize(rows)
    duration = time.time() - started

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    json_path = output_dir / f"benchmark-{timestamp}.json"
    md_path = output_dir / f"benchmark-{timestamp}.md"

    payload = {
        "model_id": args.model_id,
        "samples": args.samples,
        "seed": args.seed,
        "duration_sec": duration,
        "summary": summary,
        "rows": rows,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(md_path, args.model_id, summary, duration)

    print(f"Wrote JSON report: {json_path}")
    print(f"Wrote Markdown report: {md_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
