"""API tests with OpenAI client stubbed."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.schemas import EvidenceArtifact
from app.utils.openai_client import OpenAIClient
from app.utils.settings import get_settings

from .conftest import make_fake_chat_json


@pytest.fixture
def client_with_key(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    get_settings.cache_clear()
    return client


def test_generate_ok(client_with_key: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OpenAIClient, "chat_json", make_fake_chat_json())
    r = client_with_key.post(
        "/api/generate",
        json={
            "situation": "Running late to a meeting",
            "tone": "honest",
            "target": "manager",
            "existing_message": None,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["best_mode"] == "honest"
    assert data["best_message"]
    assert len(data["drafts"]) == 5
    assert "evidence" not in data


def test_generate_respects_requested_tone(
    client_with_key: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(OpenAIClient, "chat_json", make_fake_chat_json())
    r = client_with_key.post(
        "/api/generate",
        json={
            "situation": "Running late to a meeting",
            "tone": "absurd",
            "target": "manager",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["best_mode"] == "absurd"
    assert "time-traveling printer" in (data["best_message"] or "")


def test_generate_best_follows_tone_not_highest_judge_score(
    client_with_key: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Judge scores professional higher in fake data, but best message must match requested tone."""
    monkeypatch.setattr(OpenAIClient, "chat_json", make_fake_chat_json())
    r = client_with_key.post(
        "/api/generate",
        json={
            "situation": "Running late to a meeting",
            "tone": "honest",
            "target": "manager",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["best_mode"] == "honest"
    assert "explain the delay briefly" in (data["best_message"] or "")
    assert "formal tone" not in (data["best_message"] or "").lower()


def test_generate_no_openai_key(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Must override any value from `.env` (pydantic-settings loads the file).
    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings.cache_clear()
    r = client.post(
        "/api/generate",
        json={
            "situation": "x",
            "tone": "honest",
            "target": "manager",
        },
    )
    assert r.status_code == 503


def test_evidence_image_ok(
    client_with_key: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", "hf_test_token")
    get_settings.cache_clear()

    def _fake_illustration(*_args, **_kwargs) -> EvidenceArtifact:
        return EvidenceArtifact(
            kind="generated_image",
            caption="test caption",
            image_base64="aGVsbG8=",
            mime_type="image/png",
        )

    monkeypatch.setattr(
        "app.api.routes.generate_draft_illustration",
        _fake_illustration,
    )
    r = client_with_key.post(
        "/api/evidence/image",
        json={
            "situation": "Late to interview",
            "target": "manager",
            "draft_mode": "honest",
            "draft_text": "Sorry I was stuck in traffic.",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["caption"] == "test caption"
    assert data["image_base64"] == "aGVsbG8="


def test_evidence_image_503_without_hf_token(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", "")
    get_settings.cache_clear()
    r = client.post(
        "/api/evidence/image",
        json={
            "situation": "x",
            "target": "friend",
            "draft_mode": "honest",
            "draft_text": "hello",
        },
    )
    assert r.status_code == 503
