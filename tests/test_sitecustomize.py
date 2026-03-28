from __future__ import annotations

import unittest


class SiteCustomizeTests(unittest.TestCase):
    def test_attention_utils_falls_back_without_flash_attn(self) -> None:
        import verl.utils.attention_utils as attention_utils

        _, _, _, unpad_input = attention_utils._get_attention_functions()
        self.assertEqual(unpad_input.__module__, "verl.utils.npu_flash_attn_utils")


if __name__ == "__main__":
    unittest.main()
