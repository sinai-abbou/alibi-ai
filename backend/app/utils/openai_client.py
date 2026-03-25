"""Thin OpenAI wrapper for JSON-oriented agent calls."""

from __future__ import annotations

import json
from typing import Any, cast

from openai import OpenAI

from app.utils.settings import Settings


class OpenAIClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Placeholder avoids SDK init failure when key is unset; routes must still reject requests.
        key = settings.openai_api_key or "invalid-placeholder-not-for-production"
        self._client = OpenAI(api_key=key)

    def chat_json(self, *, system: str, user: str, temperature: float = 0.7) -> dict[str, Any]:
        """Return a JSON object from the model (response_format json_object)."""
        resp = self._client.chat.completions.create(
            model=self._settings.openai_model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content
        if not content:
            return {}
        return cast(dict[str, Any], json.loads(content))
