from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from active_perception_r1.utils.agent_loop_dataset import normalize_agent_loop_row, prepare_agent_loop_parquet
from active_perception_r1.utils.multimodal_messages import strip_none_fields_from_messages


class AgentLoopDatasetTests(unittest.TestCase):
    def test_normalize_agent_loop_row_adds_reward_model_and_strips_none_fields(self) -> None:
        row = {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "chart.png", "text": None},
                        {"type": "text", "text": "Read the inset.", "image": None},
                    ],
                }
            ],
            "ground_truth": "42",
            "data_source": "active_perception_v0",
        }

        normalized = normalize_agent_loop_row(row)

        self.assertEqual(normalized["reward_model"]["ground_truth"], "42")
        self.assertNotIn("text", normalized["prompt"][0]["content"][0])
        self.assertNotIn("image", normalized["prompt"][0]["content"][1])

    def test_prepare_agent_loop_parquet_materializes_expected_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.parquet"
            output_path = temp_path / "output.parquet"

            table = pa.Table.from_pylist(
                [
                    {
                        "prompt": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": "chart.png", "text": None},
                                    {"type": "text", "text": "Read the inset.", "image": None},
                                ],
                            }
                        ],
                        "images": ["chart.png"],
                        "ground_truth": "42",
                        "data_source": "active_perception_v0",
                        "extra_info": {"requires_zoom": True},
                    }
                ]
            )
            pq.write_table(table, input_path)

            prepare_agent_loop_parquet(input_path, output_path)

            row = pq.read_table(output_path).to_pylist()[0]
            self.assertEqual(row["reward_model"]["ground_truth"], "42")
            sanitized_prompt = strip_none_fields_from_messages(row["prompt"])
            self.assertEqual(sanitized_prompt[0]["content"][0], {"type": "image", "image": "chart.png"})
            self.assertEqual(sanitized_prompt[0]["content"][1], {"type": "text", "text": "Read the inset."})


if __name__ == "__main__":
    unittest.main()
