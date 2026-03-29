"""Risk analyzer: flags training-mode compliance issues per draft."""

from __future__ import annotations

import json
from typing import Any

from app.agents.judge_scoring import normalize_score
from app.schemas import DraftMode, MessageDraft, RiskPerDraft
from app.utils.logging import get_logger
from app.utils.openai_client import OpenAIClient
from app.utils.settings import Settings

logger = get_logger(__name__)

SYSTEM = """You evaluate fictional training messages for safety and compliance in a simulation.
You output JSON only. policy_risk is 0-10 where higher means more concerning (e.g. implies
real-world deception, forgery, or hiding misconduct). framing_ok is false if the draft does not
clearly read as fictional/training."""


def analyze_risks(
    client: OpenAIClient,
    settings: Settings,
    *,
    drafts: list[MessageDraft],
    situation: str,
    target_audience: str,
) -> dict[str, RiskPerDraft]:
    """Return risk keyed by draft mode value."""
    payload = {
        "situation": situation,
        "target_audience": target_audience,
        "drafts": [
            {
                "mode": d.mode.value,
                "text": d.text,
                "fictional_framing_present": d.fictional_framing_present,
            }
            for d in drafts
        ],
    }
    user = (
        'Analyze each draft. Return JSON: { "per_draft": [ { "mode": "...", '
        '"policy_risk": 0-10, "warnings": ["..."], "framing_ok": true/false } ] }.\n'
        f"Input: {json.dumps(payload)}"
    )
    logger.info("Risk analyzer: model=%s", settings.openai_model)
    raw = client.chat_json(system=SYSTEM, user=user, temperature=0.2)
    return _parse_risk(raw, [d.mode for d in drafts])


def _parse_risk(raw: dict[str, Any], modes: list[DraftMode]) -> dict[str, RiskPerDraft]:
    out: dict[str, RiskPerDraft] = {}
    rows = raw.get("per_draft", raw.get("drafts", []))
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
        pr = normalize_score(row.get("policy_risk"), default=5.0)
        warnings = row.get("warnings", [])
        if not isinstance(warnings, list):
            warnings = [str(warnings)]
        warnings = [str(w) for w in warnings]
        framing_ok = bool(row.get("framing_ok", True))
        out[mode.value] = RiskPerDraft(policy_risk=pr, warnings=warnings, framing_ok=framing_ok)
    # Fill missing modes with conservative defaults
    for m in modes:
        if m.value not in out:
            out[m.value] = RiskPerDraft(
                policy_risk=5,
                warnings=["missing risk analysis"],
                framing_ok=False,
            )
    return out
