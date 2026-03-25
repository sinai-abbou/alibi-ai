"""TDD tests for deterministic judge scoring and selection."""

from __future__ import annotations

import math

from app.agents.judge_scoring import composite_score, normalize_score, select_best_draft
from app.schemas import DraftMode, JudgeScores, MessageDraft, RiskPerDraft


def test_normalize_clamps() -> None:
    assert normalize_score(-1) == 0
    assert normalize_score(11) == 10
    assert normalize_score(3.3) == 3.3


def test_normalize_none_and_nan() -> None:
    assert normalize_score(None) == 0
    assert normalize_score(float("nan")) == 0


def test_composite_monotonic_in_judge() -> None:
    low = JudgeScores(plausibility=0, coherence=0, training_compliance=0)
    high = JudgeScores(plausibility=10, coherence=10, training_compliance=10)
    risk = RiskPerDraft(policy_risk=0, warnings=[], framing_ok=True)
    assert composite_score(high, risk) > composite_score(low, risk)


def test_composite_higher_risk_lowers_score() -> None:
    j = JudgeScores(plausibility=5, coherence=5, training_compliance=5)
    r0 = RiskPerDraft(policy_risk=0, warnings=[], framing_ok=True)
    r10 = RiskPerDraft(policy_risk=10, warnings=[], framing_ok=False)
    assert composite_score(j, r0) > composite_score(j, r10)


def test_select_best_simple() -> None:
    drafts = [
        MessageDraft(mode=DraftMode.HONEST, text="a"),
        MessageDraft(mode=DraftMode.ABSURD, text="b"),
    ]
    judge = {
        "honest": JudgeScores(plausibility=5, coherence=5, training_compliance=5),
        "absurd": JudgeScores(plausibility=9, coherence=9, training_compliance=9),
    }
    risk = {
        "honest": RiskPerDraft(policy_risk=0, warnings=[], framing_ok=True),
        "absurd": RiskPerDraft(policy_risk=0, warnings=[], framing_ok=True),
    }
    mode, score = select_best_draft(drafts, judge, risk)
    assert mode == DraftMode.ABSURD
    assert score is not None and score > 0


def test_select_tie_breaker_plausibility() -> None:
    drafts = [
        MessageDraft(mode=DraftMode.HONEST, text="a"),
        MessageDraft(mode=DraftMode.EXAGGERATED, text="b"),
    ]
    # Same composite; different plausibility — higher plausibility wins on tie-break.
    # honest: 0.35*8 + 0.25*5 + 0.30*5 - 0.15*1 = 5.4
    # exaggerated: 0.35*9 + 0.25*3.6 + 0.30*5 - 0.15*1 = 3.15 + 0.9 + 1.5 - 0.15 = 5.4
    judge = {
        "honest": JudgeScores(plausibility=8, coherence=5, training_compliance=5),
        "exaggerated": JudgeScores(plausibility=9, coherence=3.6, training_compliance=5),
    }
    risk = {
        "honest": RiskPerDraft(policy_risk=1, warnings=[], framing_ok=True),
        "exaggerated": RiskPerDraft(policy_risk=1, warnings=[], framing_ok=True),
    }
    c1 = composite_score(judge["honest"], risk["honest"])
    c2 = composite_score(judge["exaggerated"], risk["exaggerated"])
    assert math.isclose(c1, c2, rel_tol=0, abs_tol=1e-9)
    mode, _ = select_best_draft(drafts, judge, risk)
    assert mode == DraftMode.EXAGGERATED


def test_select_full_tie_uses_mode_order() -> None:
    """When score, plausibility, coherence match, later alphabet mode wins (higher index)."""
    drafts = [
        MessageDraft(mode=DraftMode.HONEST, text="a"),
        MessageDraft(mode=DraftMode.EXAGGERATED, text="b"),
    ]
    j = JudgeScores(plausibility=5, coherence=5, training_compliance=5)
    r = RiskPerDraft(policy_risk=1, warnings=[], framing_ok=True)
    judge = {"honest": j, "exaggerated": j}
    risk = {"honest": r, "exaggerated": r}
    mode, _ = select_best_draft(drafts, judge, risk)
    assert mode == DraftMode.HONEST


def test_select_empty() -> None:
    assert select_best_draft([], {}, {}) == (None, None)
