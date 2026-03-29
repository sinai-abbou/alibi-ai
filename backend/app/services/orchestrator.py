"""Orchestrator: RAG → Generator → Risk → Judge → tone-based selection."""

from __future__ import annotations

import uuid

from app.agents.generator import generate_drafts
from app.agents.judge import judge_drafts
from app.agents.judge_scoring import composite_score
from app.agents.risk_analyzer import analyze_risks
from app.rag.retriever import KnowledgeRetriever
from app.schemas import (
    DraftMode,
    GenerateRequest,
    GenerateResponse,
    JudgeScores,
    MessageDraft,
    RiskPerDraft,
)
from app.utils.logging import get_logger
from app.utils.openai_client import OpenAIClient
from app.utils.settings import Settings

logger = get_logger(__name__)


def _draft_for_tone(drafts: list[MessageDraft], tone: DraftMode) -> MessageDraft | None:
    for d in drafts:
        if d.mode == tone:
            return d
    return None


def run_pipeline(
    *,
    request: GenerateRequest,
    settings: Settings,
    retriever: KnowledgeRetriever,
    client: OpenAIClient,
    request_id: str | None = None,
) -> GenerateResponse:
    rid = request_id or str(uuid.uuid4())
    logger.info("[%s] pipeline: start", rid)

    q = f"{request.situation} {request.tone.value} {request.target.value}"
    rag_chunks = retriever.retrieve(q)
    logger.info("[%s] rag: retrieved %d chunks", rid, len(rag_chunks))

    drafts = generate_drafts(client, settings, request=request, rag_chunks=rag_chunks)
    if not drafts:
        logger.warning("[%s] generator: no drafts", rid)
        return GenerateResponse(
            request_id=rid,
            situation_summary=request.situation[:500],
            rag_context_used=rag_chunks,
            drafts=[],
            warnings=["Generator returned no drafts; check API key and model."],
            meta={"error": "no_drafts"},
        )

    logger.info("[%s] generator: %d drafts", rid, len(drafts))

    risk = analyze_risks(
        client,
        settings,
        drafts=drafts,
        situation=request.situation,
        target_audience=request.target.value,
    )
    logger.info("[%s] risk: analyzed %d modes", rid, len(risk))

    judge_scores = judge_drafts(
        client,
        settings,
        drafts=drafts,
        situation=request.situation,
        target_audience=request.target.value,
    )
    logger.info("[%s] judge: scores for %d modes", rid, len(judge_scores))

    selected = _draft_for_tone(drafts, request.tone)
    best_mode: DraftMode | None = None
    best_msg: str | None = None
    composite: float | None = None

    if selected is not None:
        best_mode = selected.mode
        best_msg = selected.text
        j = judge_scores.get(selected.mode.value) or JudgeScores(
            plausibility=0, coherence=0, training_compliance=0
        )
        rk = risk.get(selected.mode.value) or RiskPerDraft(
            policy_risk=5, warnings=["missing judge/risk for mode"], framing_ok=False
        )
        composite = composite_score(j, rk)
        logger.info(
            "[%s] selection: user tone=%s (judge not used for selection)",
            rid,
            request.tone.value,
        )
    else:
        logger.warning(
            "[%s] no draft for requested tone=%s",
            rid,
            request.tone.value,
        )

    warnings: list[str] = []
    if selected is None:
        warnings.append(
            f"No draft was produced for the selected tone '{request.tone.value}'. "
            "Try generating again."
        )
    for d in drafts:
        r = risk.get(d.mode.value)
        if r:
            warnings.extend(r.warnings)
            if not r.framing_ok:
                warnings.append(f"Framing may be unclear for mode={d.mode.value}.")

    logger.info("[%s] pipeline: done best_mode=%s", rid, best_mode)

    return GenerateResponse(
        request_id=rid,
        situation_summary=request.situation[:500],
        rag_context_used=rag_chunks,
        drafts=drafts,
        judge_scores=judge_scores,
        risk=risk,
        best_mode=best_mode,
        best_message=best_msg,
        composite_score=composite,
        warnings=warnings,
        meta={"model": settings.openai_model},
    )
