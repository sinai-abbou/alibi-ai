"""Lightweight RAG: embed knowledge cards and retrieve by cosine similarity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from app.utils.logging import get_logger
from app.utils.settings import Settings

logger = get_logger(__name__)


@dataclass(frozen=True)
class KnowledgeCard:
    id: str
    title: str
    body: str
    tags: list[str]

    def as_text(self) -> str:
        tags = ", ".join(self.tags) if self.tags else ""
        return f"{self.title}\n{self.body}\nTags: {tags}"


def _cards_fingerprint(cards: list[KnowledgeCard]) -> str:
    raw = json.dumps([c.__dict__ for c in cards], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


class KnowledgeRetriever:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: SentenceTransformer | None = None
        self._cards: list[KnowledgeCard] = []
        self._embeddings: np.ndarray | None = None

    def _load_cards(self) -> list[KnowledgeCard]:
        path: Path = self._settings.knowledge_cards_path
        data = json.loads(path.read_text(encoding="utf-8"))
        out: list[KnowledgeCard] = []
        for row in data:
            out.append(
                KnowledgeCard(
                    id=str(row["id"]),
                    title=str(row["title"]),
                    body=str(row["body"]),
                    tags=list(row.get("tags", [])),
                )
            )
        return out

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self._settings.embedding_model)
            self._model = SentenceTransformer(self._settings.embedding_model)
        return self._model

    def _try_load_cache(self, fp: str) -> np.ndarray | None:
        cache_path = self._settings.embeddings_cache_path
        if not cache_path.is_file():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") != fp:
                return None
            arr = np.array(payload["embeddings"], dtype=np.float32)
            return arr
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def _save_cache(self, fp: str, embeddings: np.ndarray) -> None:
        cache_path = self._settings.embeddings_cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "fingerprint": fp,
            "model": self._settings.embedding_model,
            "embeddings": embeddings.tolist(),
        }
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        logger.info("Wrote embeddings cache to %s", cache_path)

    def warm(self) -> None:
        """Preload cards and embeddings (call at startup)."""
        self._cards = self._load_cards()
        fp = _cards_fingerprint(self._cards)
        cached = self._try_load_cache(fp)
        if cached is not None and len(cached) == len(self._cards):
            self._embeddings = cached
            logger.info("Loaded %d card embeddings from cache", len(self._cards))
            return
        model = self._get_model()
        texts = [c.as_text() for c in self._cards]
        enc = model.encode(texts, normalize_embeddings=True)
        self._embeddings = np.array(enc, dtype=np.float32)
        self._save_cache(fp, self._embeddings)

    def retrieve(self, query: str, k: int | None = None) -> list[str]:
        if not self._cards or self._embeddings is None:
            self.warm()
        assert self._embeddings is not None
        kk = k if k is not None else self._settings.rag_top_k
        model = self._get_model()
        q = np.array(model.encode([query], normalize_embeddings=True), dtype=np.float32)[0]
        mat = self._embeddings
        sims = mat @ q
        idx = np.argsort(-sims)[:kk]
        return [self._cards[int(i)].as_text() for i in idx]
