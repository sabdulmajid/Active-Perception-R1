#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from active_perception_r1.sim.self_driving_lab import generate_policy_sweep_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic self-driving active-perception policy sweeps.")
    parser.add_argument("--output-dir", default="results", help="Directory to store JSON and Markdown outputs.")
    parser.add_argument("--scenes-per-combo", type=int, default=120, help="Scenes per task-condition-seed combination.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23], help="Random seeds.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = generate_policy_sweep_report(scenes_per_combo=args.scenes_per_combo, seeds=args.seeds)

    json_path = output_dir / "self_driving_policy_sweep.json"
    md_path = output_dir / "self_driving_policy_sweep.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(report["markdown_summary"], encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

