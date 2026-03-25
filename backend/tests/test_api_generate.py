"""API tests with OpenAI client stubbed."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.utils.openai_client import OpenAIClient
from app.utils.settings import get_settings

from .conftest import make_fake_chat_json


@pytest.fixture
def client_with_key(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    get_settings.cache_clear()
    return client


def test_generate_ok(client_with_key: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OpenAIClient, "chat_json", make_fake_chat_json(include_evidence_plan=False))
    r = client_with_key.post(
        "/api/generate",
        json={
            "situation": "Running late to a meeting",
            "tone": "polite",
            "target": "manager",
            "existing_message": None,
            "generate_evidence": False,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["best_mode"] == "honest"
    assert data["best_message"]
    assert len(data["drafts"]) >= 2


def test_generate_evidence_flags(
    client_with_key: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(OpenAIClient, "chat_json", make_fake_chat_json(include_evidence_plan=True))
    r = client_with_key.post(
        "/api/generate",
        json={
            "situation": "Running late to a meeting",
            "tone": "polite",
            "target": "manager",
            "generate_evidence": True,
        },
    )
    assert r.status_code == 200
    data = r.json()
    for art in data.get("evidence", []):
        assert art.get("synthetic") is True
        assert art.get("non_verifiable") is True


def test_generate_no_openai_key(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Must override any value from `.env` (pydantic-settings loads the file).
    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings.cache_clear()
    r = client.post(
        "/api/generate",
        json={
            "situation": "x",
            "tone": "t",
            "target": "y",
            "generate_evidence": False,
        },
    )
    assert r.status_code == 503
