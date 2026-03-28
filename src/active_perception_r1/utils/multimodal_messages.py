from __future__ import annotations

from typing import Any


def strip_none_fields_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_messages: list[dict[str, Any]] = []
    for message in messages:
        normalized_message = dict(message)
        content = normalized_message.get("content")
        if isinstance(content, list):
            normalized_content: list[Any] = []
            for item in content:
                if not isinstance(item, dict):
                    normalized_content.append(item)
                    continue
                normalized_content.append({key: value for key, value in item.items() if value is not None})
            normalized_message["content"] = normalized_content
        elif isinstance(content, dict):
            normalized_message["content"] = {key: value for key, value in content.items() if value is not None}
        normalized_messages.append(normalized_message)
    return normalized_messages
