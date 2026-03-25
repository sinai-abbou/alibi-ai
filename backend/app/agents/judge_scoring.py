"""Deterministic scoring helpers for Judge + best-draft selection (TDD core)."""

from __future__ import annotations

from app.schemas import DraftMode, JudgeScores, MessageDraft, RiskPerDraft


def normalize_score(value: object | None, *, default: float = 0.0) -> float:
    """Clamp a score to [0, 10]; None, NaN, or non-numeric maps to default."""
    if value is None:
        return default
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    return max(0.0, min(10.0, v))


def composite_score(judge: JudgeScores, risk: RiskPerDraft) -> float:
    """
    Higher is better. Uses judge dimensions minus policy risk.

    Weights sum to 1.0 on the judge side; risk is subtracted with weight 0.15.
    """
    p = normalize_score(judge.plausibility)
    c = normalize_score(judge.coherence)
    t = normalize_score(judge.training_compliance)
    r = normalize_score(risk.policy_risk)
    return 0.35 * p + 0.25 * c + 0.30 * t - 0.15 * r


def _mode_key(mode: DraftMode) -> str:
    return mode.value


def select_best_draft(
    drafts: list[MessageDraft],
    judge_scores: dict[str, JudgeScores],
    risk: dict[str, RiskPerDraft],
) -> tuple[DraftMode | None, float | None]:
    """
    Pick the draft with highest composite score.

    Ties: higher plausibility, then higher coherence, then mode order:
    honest < exaggerated < absurd < hypothetical (lexicographic by value).
    """
    if not drafts:
        return None, None

    mode_order = sorted([m.value for m in DraftMode], key=lambda x: x)

    best_mode: DraftMode | None = None
    best_score: float | None = None
    best_tuple: tuple[float, float, float, int] | None = None

    for d in drafts:
        key = _mode_key(d.mode)
        j = judge_scores.get(key) or JudgeScores(plausibility=0, coherence=0, training_compliance=0)
        rk = risk.get(key) or RiskPerDraft(policy_risk=10, warnings=[], framing_ok=False)
        score = composite_score(j, rk)
        p = normalize_score(j.plausibility)
        c = normalize_score(j.coherence)
        tie = (score, p, c, mode_order.index(key))
        if best_tuple is None or tie > best_tuple:
            best_tuple = tie
            best_score = score
            best_mode = d.mode

    return best_mode, best_score
