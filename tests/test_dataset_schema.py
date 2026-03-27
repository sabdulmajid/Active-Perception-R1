from __future__ import annotations

import unittest

from active_perception_r1.utils.dataset_schema import DISABLED_IMAGE_KEY, resolve_verl_image_key_from_row


class DatasetSchemaTests(unittest.TestCase):
    def test_structured_image_content_disables_image_key(self) -> None:
        row = {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "path/to/image.png"},
                        {"type": "text", "text": "Question"},
                    ],
                }
            ]
        }

        resolved = resolve_verl_image_key_from_row(row)
        self.assertEqual(resolved, DISABLED_IMAGE_KEY)

    def test_placeholder_prompts_keep_requested_image_key(self) -> None:
        row = {
            "prompt": [
                {
                    "role": "user",
                    "content": "<image> Read the value.",
                }
            ]
        }

        resolved = resolve_verl_image_key_from_row(row, requested_image_key="images")
        self.assertEqual(resolved, "images")


if __name__ == "__main__":
    unittest.main()
