from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from active_perception_r1.utils.multimodal_messages import strip_none_fields_from_messages


def normalize_agent_loop_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    prompt = normalized.get("prompt")
    if isinstance(prompt, list):
        normalized["prompt"] = strip_none_fields_from_messages(prompt)

    reward_model = dict(normalized.get("reward_model") or {})
    if "ground_truth" not in reward_model and "ground_truth" in normalized:
        reward_model["ground_truth"] = normalized.get("ground_truth")
    normalized["reward_model"] = reward_model
    return normalized


def prepare_agent_loop_parquet(input_path: str | Path, output_path: str | Path) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = pq.read_table(input_path).to_pylist()
    normalized_rows = [normalize_agent_loop_row(row) for row in rows]
    table = pa.Table.from_pylist(normalized_rows)
    pq.write_table(table, output_path)
    return output_path


def default_prepared_output_path(input_path: str | Path, output_dir: str | Path) -> Path:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    return output_dir / f"{input_path.stem}.agent_loop.parquet"


def _main() -> int:
    parser = argparse.ArgumentParser(description="Prepare parquet datasets for verl agent-loop reward expectations.")
    parser.add_argument("--input", required=True, help="Source parquet path.")
    parser.add_argument(
        "--output",
        help="Destination parquet path. If omitted, --output-dir must be provided.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for prepared parquet output. The filename is derived from the input basename.",
    )
    args = parser.parse_args()

    if not args.output and not args.output_dir:
        raise SystemExit("Either --output or --output-dir is required.")

    output_path = Path(args.output) if args.output else default_prepared_output_path(args.input, args.output_dir)
    prepared_path = prepare_agent_loop_parquet(args.input, output_path)
    print(prepared_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
