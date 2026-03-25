"""Generator agent: produces 2–3 fictional training message drafts."""

from __future__ import annotations

from typing import Any

from app.schemas import DraftMode, GenerateRequest, MessageDraft
from app.utils.logging import get_logger
from app.utils.openai_client import OpenAIClient
from app.utils.settings import Settings

logger = get_logger(__name__)

SYSTEM = """You are a communication training assistant. You ONLY produce fictional examples
for classroom/simulation use. Outputs must clearly read as hypothetical training material,
not as factual claims or real-world instructions. Never encourage deception or forgery."""


def _parse_drafts(raw: dict[str, Any]) -> list[MessageDraft]:
    out: list[MessageDraft] = []
    for row in raw.get("drafts", []):
        mode_str = str(row.get("mode", "")).lower()
        try:
            mode = DraftMode(mode_str)
        except ValueError:
            continue
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        framing = bool(row.get("fictional_framing_present", True))
        out.append(MessageDraft(mode=mode, text=text, fictional_framing_present=framing))
    return out


def generate_drafts(
    client: OpenAIClient,
    settings: Settings,
    *,
    request: GenerateRequest,
    rag_chunks: list[str],
) -> list[MessageDraft]:
    rag_text = "\n---\n".join(rag_chunks) if rag_chunks else "(no retrieved context)"
    user_parts = [
        "Situation (training context):",
        request.situation,
        f"Tone: {request.tone}",
        f"Target audience: {request.target}",
        "Retrieved knowledge snippets (may be partial):",
        rag_text,
    ]
    if request.existing_message:
        user_parts.extend(
            [
                "Optional existing user draft to rewrite as a fictional training example:",
                request.existing_message,
            ]
        )
    user_parts.append(
        "Return JSON with key 'drafts': an array of 3 objects with keys "
        "mode (one of: honest, exaggerated, absurd, hypothetical), "
        "text (short message), fictional_framing_present (boolean). "
        "Each text must clearly be a fictional training example, not a real claim."
    )
    user = "\n".join(user_parts)
    logger.info("Generator: calling OpenAI model=%s", settings.openai_model)
    raw = client.chat_json(system=SYSTEM, user=user, temperature=0.8)
    drafts = _parse_drafts(raw)
    if len(drafts) < 2:
        logger.warning(
            "Generator: expected >=2 drafts, got %d; retrying with stricter prompt",
            len(drafts),
        )
        raw = client.chat_json(
            system=SYSTEM,
            user=user + "\nYou MUST return exactly 3 drafts in the array.",
            temperature=0.6,
        )
        drafts = _parse_drafts(raw)
    return drafts[:4]
