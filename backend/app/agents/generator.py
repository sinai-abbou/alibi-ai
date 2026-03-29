"""Generator agent: produces one fictional training draft per tone mode."""

from __future__ import annotations

from typing import Any

from app.schemas import (
    DRAFT_MODES_ORDER,
    DraftMode,
    GenerateRequest,
    MessageDraft,
    TargetAudience,
)
from app.utils.logging import get_logger
from app.utils.openai_client import OpenAIClient
from app.utils.settings import Settings

logger = get_logger(__name__)

SYSTEM = """You are a communication training assistant. You ONLY produce fictional examples
for classroom/simulation use. Outputs must clearly read as hypothetical training material,
not as factual claims or real-world instructions. Never encourage deception or forgery."""

# Register and structure hints per audience (must align with TargetAudience enum).
_AUDIENCE_INSTRUCTIONS: dict[TargetAudience, str] = {
    TargetAudience.FRIEND: (
        "Audience: friend — use a casual, relaxed register; short messages; informal vocabulary; "
        "warm and direct; contractions OK; avoid corporate jargon."
    ),
    TargetAudience.FAMILY: (
        "Audience: family — warm, caring tone; emotional openness appropriate to kin; "
        "clear and supportive; not stiff or corporate."
    ),
    TargetAudience.COWORKER: (
        "Audience: coworker — polite and collegial but informal; workplace context; "
        "clear and cooperative; no heavy hierarchy language."
    ),
    TargetAudience.MANAGER: (
        "Audience: manager — respectful, professional, concise; acknowledge impact; "
        "appropriate deference without being obsequious."
    ),
    TargetAudience.CLIENT: (
        "Audience: client — very professional, clear, and polished; customer-facing register; "
        "no slang; prioritize clarity and courtesy."
    ),
    TargetAudience.PARTNER: (
        "Audience: romantic partner — personal, emotionally attuned, intimate but respectful; "
        "first-person feelings; softer, closer register than friends."
    ),
    TargetAudience.TEACHER_PROFESSOR: (
        "Audience: teacher or professor — respectful, slightly formal student–educator register; "
        "clear and polite; no slang; appropriate titles if natural in context."
    ),
    TargetAudience.FORMAL_OFFICIAL: (
        "Audience: generic formal / official — neutral formal register suitable for institutions; "
        "structured, impersonal where appropriate; no colloquialisms."
    ),
}


def _parse_drafts(raw: dict[str, Any]) -> list[MessageDraft]:
    out: list[MessageDraft] = []
    for row in raw.get("drafts", []):
        if not isinstance(row, dict):
            continue
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


def _merge_unique_by_mode(drafts: list[MessageDraft]) -> dict[DraftMode, MessageDraft]:
    by: dict[DraftMode, MessageDraft] = {}
    for d in drafts:
        if d.mode not in by:
            by[d.mode] = d
    return by


def _order_drafts(by: dict[DraftMode, MessageDraft]) -> list[MessageDraft]:
    return [by[m] for m in DRAFT_MODES_ORDER if m in by]


def generate_drafts(
    client: OpenAIClient,
    settings: Settings,
    *,
    request: GenerateRequest,
    rag_chunks: list[str],
) -> list[MessageDraft]:
    rag_text = "\n---\n".join(rag_chunks) if rag_chunks else "(no retrieved context)"
    audience_key = request.target
    audience_line = _AUDIENCE_INSTRUCTIONS.get(
        audience_key,
        f"Audience: {audience_key.value}.",
    )
    user_parts = [
        "Situation (training context):",
        request.situation,
        f"User's selected tone (one draft must match this mode): {request.tone.value}",
        f"User's selected recipient: {audience_key.value}",
        audience_line,
        "Apply BOTH the tone mode (honest/exaggerated/absurd/professional/emotional) AND the "
        "audience instructions above to vocabulary, formality, length, and structure for "
        "every draft.",
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
        "Return JSON with key 'drafts': an array of exactly 5 objects, one per mode, "
        "with keys mode, text, fictional_framing_present (boolean). "
        "Modes (each exactly once): "
        "honest — straightforward; "
        "exaggerated — heightened drama; "
        "absurd — surreal humor but still fictional training; "
        "professional — polished, workplace-appropriate register; "
        "emotional — warmer, feelings-forward, still appropriate. "
        "Each draft must reflect the selected recipient's register AND its mode. "
        "Each text must clearly be a fictional training example, not a real claim."
    )
    user = "\n".join(user_parts)
    logger.info("Generator: calling OpenAI model=%s", settings.openai_model)
    raw = client.chat_json(system=SYSTEM, user=user, temperature=0.8)
    merged = _merge_unique_by_mode(_parse_drafts(raw))
    missing = [m for m in DRAFT_MODES_ORDER if m not in merged]
    if missing:
        logger.warning(
            "Generator: missing modes after first call: %s; retrying",
            [m.value for m in missing],
        )
        retry_user = (
            user
            + "\n\nYou MUST include exactly one draft for each of these missing modes: "
            + ", ".join(m.value for m in missing)
            + ". Return the full 'drafts' array with all 5 modes."
        )
        raw2 = client.chat_json(system=SYSTEM, user=retry_user, temperature=0.6)
        for d in _parse_drafts(raw2):
            if d.mode not in merged:
                merged[d.mode] = d
        missing2 = [m for m in DRAFT_MODES_ORDER if m not in merged]
        if missing2:
            logger.warning(
                "Generator: still missing modes after retry: %s",
                [m.value for m in missing2],
            )
    return _order_drafts(merged)
