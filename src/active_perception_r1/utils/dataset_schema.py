from __future__ import annotations

import argparse
from typing import Any


DISABLED_IMAGE_KEY = "__disabled_images__"


def prompt_uses_structured_image_content(prompt: Any) -> bool:
    if not isinstance(prompt, list):
        return False
    for message in prompt:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                return True
    return False


def prompt_uses_image_placeholders(prompt: Any) -> bool:
    if not isinstance(prompt, list):
        return False
    for message in prompt:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and "<image>" in content:
            return True
    return False


def resolve_verl_image_key_from_row(
    row: dict[str, Any],
    *,
    prompt_key: str = "prompt",
    requested_image_key: str = "images",
    disabled_image_key: str = DISABLED_IMAGE_KEY,
) -> str:
    prompt = row.get(prompt_key)
    if prompt_uses_structured_image_content(prompt):
        return disabled_image_key
    if prompt_uses_image_placeholders(prompt):
        return requested_image_key
    return disabled_image_key


def resolve_verl_image_key_from_parquet(
    parquet_path: str,
    *,
    prompt_key: str = "prompt",
    requested_image_key: str = "images",
    disabled_image_key: str = DISABLED_IMAGE_KEY,
) -> str:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=[prompt_key])
    if table.num_rows == 0:
        return requested_image_key

    row = table.slice(0, 1).to_pylist()[0]
    return resolve_verl_image_key_from_row(
        row,
        prompt_key=prompt_key,
        requested_image_key=requested_image_key,
        disabled_image_key=disabled_image_key,
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Resolve the correct verl image_key for a parquet dataset.")
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--requested-image-key", default="images")
    parser.add_argument("--disabled-image-key", default=DISABLED_IMAGE_KEY)
    args = parser.parse_args()

    resolved = resolve_verl_image_key_from_parquet(
        args.parquet_path,
        prompt_key=args.prompt_key,
        requested_image_key=args.requested_image_key,
        disabled_image_key=args.disabled_image_key,
    )
    print(resolved)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
