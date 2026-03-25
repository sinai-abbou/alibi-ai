"""RAG retriever sanity tests (embedding model stubbed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.rag.retriever import KnowledgeRetriever
from app.utils.settings import Settings


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    cards = tmp_path / "knowledge_cards.json"
    cards.write_text(
        '[{"id":"a","title":"T","body":"hello world","tags":["x"]}]',
        encoding="utf-8",
    )
    cache = tmp_path / "embeddings_cache.json"

    def fake_st(_model_name: str) -> object:
        class _Fake:
            def encode(self, texts: list[str] | str, normalize_embeddings: bool = True) -> object:
                if isinstance(texts, str):
                    texts = [texts]
                n = len(texts)
                return np.ones((n, 8), dtype=np.float32)

        return _Fake()

    monkeypatch.setattr("app.rag.retriever.SentenceTransformer", fake_st)
    base = Settings()
    return base.model_copy(
        update={
            "knowledge_cards_path": cards,
            "embeddings_cache_path": cache,
        }
    )


def test_retrieve_top_k(settings: Settings) -> None:
    r = KnowledgeRetriever(settings)
    r.warm()
    out = r.retrieve("hello", k=1)
    assert len(out) == 1
    assert "hello" in out[0].lower() or "T" in out[0]
