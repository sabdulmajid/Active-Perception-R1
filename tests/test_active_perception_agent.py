from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_TRAIN_ROOT = REPO_ROOT / ".vendor_train"
if str(VENDOR_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_TRAIN_ROOT))

IMPORT_ERROR = None
try:
    from active_perception_r1.rollout.active_perception_agent import ActivePerceptionZoomAgentLoop
except Exception as exc:  # pragma: no cover - exercised as a skip path in minimal envs
    ActivePerceptionZoomAgentLoop = None
    IMPORT_ERROR = exc


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def _flatten_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
            elif item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif item.get("type") == "image":
                parts.append("<image>")
        return "".join(parts)
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return str(content)


class FakeTokenizer:
    pad_token = ""

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(int(token_id)) for token_id in token_ids if int(token_id) > 0)

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        tools=None,
        return_dict: bool = False,
        **kwargs,
    ):
        del tools, kwargs
        text = "".join(f"<{message['role']}>" + _flatten_content(message["content"]) for message in messages)
        if add_generation_prompt:
            text += "<assistant>"
        if not tokenize:
            return text
        token_ids = self.encode(text)
        if return_dict:
            return {"input_ids": [token_ids]}
        return token_ids


class FakeProcessor(FakeTokenizer):
    def __init__(self) -> None:
        self.image_processor = SimpleNamespace(patch_size=14)

    def __call__(
        self,
        *,
        text: list[str],
        images=None,
        videos=None,
        video_metadata=None,
        return_tensors: str = "pt",
        do_sample_frames: bool = False,
    ) -> dict[str, list[list[int]]]:
        del images, videos, video_metadata, return_tensors, do_sample_frames
        return {"input_ids": [self.encode(text[0])]}


class FakeDataset:
    @staticmethod
    async def process_vision_info(messages, image_patch_size, config):
        del image_patch_size, config
        images = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image" and item.get("image") is not None:
                    images.append(item["image"])
        return images, None


class StrictFakeDataset(FakeDataset):
    @staticmethod
    async def process_vision_info(messages, image_patch_size, config):
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    if "image" in item:
                        raise AssertionError("text content still carries an image field")
                if item.get("type") == "image":
                    if "text" in item:
                        raise AssertionError("image content still carries a text field")
        return await FakeDataset.process_vision_info(messages, image_patch_size, config)


class FakeTokenOutput:
    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids
        self.log_probs = [-0.1] * len(token_ids)
        self.routed_experts = None
        self.num_preempted = None
        self.extra_fields = {}


class FakeServerManager:
    def __init__(self, tokenizer: FakeTokenizer, responses: list[str]) -> None:
        self.tokenizer = tokenizer
        self.responses = list(responses)

    async def generate(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict,
        image_data=None,
        video_data=None,
    ) -> FakeTokenOutput:
        del request_id, prompt_ids, sampling_params, image_data, video_data
        if not self.responses:
            raise AssertionError("No fake responses remaining for agent loop test.")
        return FakeTokenOutput(self.tokenizer.encode(self.responses.pop(0)))


@unittest.skipUnless(IMPORT_ERROR is None, f"verl agent loop deps unavailable: {IMPORT_ERROR}")
class ActivePerceptionAgentLoopTests(unittest.TestCase):
    def assert_bbox_close(
        self,
        actual: tuple[float, float, float, float],
        expected: tuple[float, float, float, float],
    ) -> None:
        for actual_value, expected_value in zip(actual, expected, strict=True):
            self.assertAlmostEqual(actual_value, expected_value, places=6)

    def test_agent_executes_zoom_and_reinjects_crop(self) -> None:
        tokenizer = FakeTokenizer()
        processor = FakeProcessor()
        trainer_config = SimpleNamespace(
            config=AttrDict(
                {
                    "actor_rollout_ref": AttrDict(
                        {
                            "rollout": AttrDict(
                                {
                                    "prompt_length": 4096,
                                    "response_length": 4096,
                                    "multi_turn": AttrDict(
                                        {
                                            "max_user_turns": 3,
                                            "max_assistant_turns": 4,
                                            "max_parallel_calls": 1,
                                            "max_tool_response_length": 256,
                                            "tool_response_truncate_side": "middle",
                                            "tool_config_path": None,
                                            "format": "hermes",
                                            "interaction_config_path": None,
                                        }
                                    ),
                                }
                            ),
                            "model": AttrDict({}),
                        }
                    )
                }
            )
        )
        data_config = SimpleNamespace(config=AttrDict({}))
        initial_image = Image.new("RGB", (600, 600), color="white")
        server = FakeServerManager(
            tokenizer,
            responses=[
                (
                    "<think>Need a closer look."
                    '<zoom_roi x0="0.50" y0="0.50" x1="0.75" y1="0.75" />'
                    "</think><answer>wrong</answer>"
                ),
                "<think>The cropped value is clear now.</think><answer>42</answer>",
            ],
        )

        async def run_agent():
            agent = ActivePerceptionZoomAgentLoop(
                trainer_config=trainer_config,
                server_manager=server,
                tokenizer=tokenizer,
                processor=processor,
                dataset_cls=FakeDataset,
                data_config=data_config,
                max_zoom_calls=3,
                min_relative_area=0.02,
                max_relative_area=0.65,
            )
            return await agent.run(
                sampling_params={},
                raw_prompt=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": initial_image},
                            {"type": "text", "text": "What value is in the highlighted inset?"},
                        ],
                    }
                ],
                extra_info={
                    "requires_zoom": True,
                    "image_size": {"width": 1000, "height": 1000},
                    "relevant_regions": [{"label": "target", "bbox": [0.5, 0.6, 0.65, 0.75], "weight": 1.0}],
                },
            )

        output = asyncio.run(run_agent())

        trace = output.extra_fields["active_tool_trace"][0]
        self.assertEqual(trace["status"], "zoom_executed")
        self.assert_bbox_close(trace["executed_bbox_norm"], (0.5, 0.5, 0.75, 0.75))
        self.assertEqual(output.extra_fields["zoom_call_count"], 1)
        self.assertEqual(len(output.multi_modal_data["images"]), 2)
        self.assertEqual(output.multi_modal_data["images"][-1].size, (150, 150))
        self.assertIn('matched_region="target"', output.extra_fields["executed_observation_tokens"][0])
        self.assertIn("<answer>42</answer>", tokenizer.decode(output.response_ids))
        self.assertIn(0, output.response_mask)

    def test_agent_normalizes_parquet_style_multimodal_rows(self) -> None:
        tokenizer = FakeTokenizer()
        processor = FakeProcessor()
        trainer_config = SimpleNamespace(
            config=AttrDict(
                {
                    "actor_rollout_ref": AttrDict(
                        {
                            "rollout": AttrDict(
                                {
                                    "prompt_length": 4096,
                                    "response_length": 4096,
                                    "multi_turn": AttrDict(
                                        {
                                            "max_user_turns": 3,
                                            "max_assistant_turns": 4,
                                            "max_parallel_calls": 1,
                                            "max_tool_response_length": 256,
                                            "tool_response_truncate_side": "middle",
                                            "tool_config_path": None,
                                            "format": "hermes",
                                            "interaction_config_path": None,
                                        }
                                    ),
                                }
                            ),
                            "model": AttrDict({}),
                        }
                    )
                }
            )
        )
        data_config = SimpleNamespace(config=AttrDict({}))
        initial_image = Image.new("RGB", (600, 600), color="white")
        server = FakeServerManager(tokenizer, responses=["<answer>42</answer>"])

        async def run_agent():
            agent = ActivePerceptionZoomAgentLoop(
                trainer_config=trainer_config,
                server_manager=server,
                tokenizer=tokenizer,
                processor=processor,
                dataset_cls=StrictFakeDataset,
                data_config=data_config,
                max_zoom_calls=3,
                min_relative_area=0.02,
                max_relative_area=0.65,
            )
            return await agent.run(
                sampling_params={},
                raw_prompt=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": initial_image, "text": None},
                            {
                                "type": "text",
                                "text": "Read the inset value.",
                                "image": None,
                            },
                        ],
                    }
                ],
                extra_info={"image_size": {"width": 600, "height": 600}},
            )

        output = asyncio.run(run_agent())

        self.assertEqual(len(output.multi_modal_data["images"]), 1)
        self.assertIn("<answer>42</answer>", tokenizer.decode(output.response_ids))


if __name__ == "__main__":
    unittest.main()
