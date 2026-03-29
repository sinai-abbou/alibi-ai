"""Pytest fixtures."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.main import app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """TestClient with RAG embedding model stubbed (no download)."""

    def fake_sentence_transformer(_model_name: str) -> Any:
        class _Fake:
            def encode(self, texts: list[str] | str, normalize_embeddings: bool = True) -> Any:
                if isinstance(texts, str):
                    texts = [texts]
                n = len(texts)
                return np.ones((n, 8), dtype=np.float32)

        return _Fake()

    monkeypatch.setattr(
        "app.rag.retriever.SentenceTransformer",
        fake_sentence_transformer,
    )
    with TestClient(app) as test_client:
        yield test_client


def make_fake_chat_json() -> Callable[..., dict[str, Any]]:
    responses: list[dict[str, Any]] = [
        {
            "drafts": [
                {
                    "mode": "honest",
                    "text": "Fictional training example: I might explain the delay briefly.",
                    "fictional_framing_present": True,
                },
                {
                    "mode": "exaggerated",
                    "text": "Fictional training example: the delay was enormous.",
                    "fictional_framing_present": True,
                },
                {
                    "mode": "absurd",
                    "text": "Fictional training example: a time-traveling printer caused it.",
                    "fictional_framing_present": True,
                },
                {
                    "mode": "professional",
                    "text": (
                        "Fictional training example: I regret the inconvenience in a formal tone."
                    ),
                    "fictional_framing_present": True,
                },
                {
                    "mode": "emotional",
                    "text": "Fictional training example: I feel terrible about letting you down.",
                    "fictional_framing_present": True,
                },
            ]
        },
        {
            "per_draft": [
                {
                    "mode": m,
                    "policy_risk": 1,
                    "warnings": [],
                    "framing_ok": True,
                }
                for m in (
                    "honest",
                    "exaggerated",
                    "absurd",
                    "professional",
                    "emotional",
                )
            ]
        },
        {
            "scores": [
                {"mode": "honest", "plausibility": 8, "coherence": 8, "training_compliance": 9},
                {
                    "mode": "exaggerated",
                    "plausibility": 6,
                    "coherence": 7,
                    "training_compliance": 8,
                },
                {"mode": "absurd", "plausibility": 4, "coherence": 6, "training_compliance": 9},
                {
                    "mode": "professional",
                    "plausibility": 9,
                    "coherence": 9,
                    "training_compliance": 10,
                },
                {"mode": "emotional", "plausibility": 7, "coherence": 8, "training_compliance": 9},
            ]
        },
    ]
    idx = {"n": 0}

    def _chat_json(
        _self: Any,
        *,
        system: str,
        user: str,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        i = idx["n"]
        idx["n"] += 1
        if i >= len(responses):
            return {}
        return responses[i]

    return _chat_json
