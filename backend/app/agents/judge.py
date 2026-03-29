"""Judge agent: scores narrative plausibility/coherence and training compliance."""

from __future__ import annotations

from typing import Any

from app.agents.judge_scoring import normalize_score
from app.schemas import DraftMode, JudgeScores, MessageDraft
from app.utils.logging import get_logger
from app.utils.openai_client import OpenAIClient
from app.utils.settings import Settings

logger = get_logger(__name__)

SYSTEM = """You are a communication training judge. Score each fictional message draft on:
- plausibility (0-10): narrative believability within the scenario
- coherence (0-10): internal consistency
- training_compliance (0-10): clearly framed as fictional training, not as real proof
Return JSON only."""


def judge_drafts(
    client: OpenAIClient,
    settings: Settings,
    *,
    drafts: list[MessageDraft],
    situation: str,
    target_audience: str,
) -> dict[str, JudgeScores]:
    import json

    payload = {
        "situation": situation,
        "target_audience": target_audience,
        "drafts": [{"mode": d.mode.value, "text": d.text} for d in drafts],
    }
    user = (
        "Scores must account for whether each draft fits the stated recipient "
        "(register, appropriateness). "
        'Return JSON: { "scores": [ { "mode": "honest|exaggerated|absurd|professional|emotional", '
        '"plausibility": 0-10, "coherence": 0-10, "training_compliance": 0-10 } ] }.\n'
        f"Input: {json.dumps(payload)}"
    )
    logger.info("Judge: model=%s", settings.openai_model)
    raw = client.chat_json(system=SYSTEM, user=user, temperature=0.2)
    return _parse_scores(raw)


def _parse_scores(raw: dict[str, Any]) -> dict[str, JudgeScores]:
    out: dict[str, JudgeScores] = {}
    rows = raw.get("scores", raw.get("per_draft", []))
    if not isinstance(rows, list):
        rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        mode_str = str(row.get("mode", "")).lower()
        try:
            mode = DraftMode(mode_str)
        except ValueError:
            continue
        out[mode.value] = JudgeScores(
            plausibility=normalize_score(row.get("plausibility")),
            coherence=normalize_score(row.get("coherence")),
            training_compliance=normalize_score(row.get("training_compliance")),
        )
    return out
