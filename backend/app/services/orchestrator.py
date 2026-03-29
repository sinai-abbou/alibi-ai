"""Orchestrator: RAG → Generator → Risk → Judge → selection → optional Evidence."""

from __future__ import annotations

import uuid

from app.agents.generator import generate_drafts
from app.agents.judge import judge_drafts
from app.agents.judge_scoring import select_best_draft
from app.agents.risk_analyzer import analyze_risks
from app.rag.retriever import KnowledgeRetriever
from app.schemas import EvidenceArtifact, EvidencePlanItem, GenerateRequest, GenerateResponse
from app.services.evidence import generate_evidence_bundle
from app.utils.logging import get_logger
from app.utils.openai_client import OpenAIClient
from app.utils.settings import Settings

logger = get_logger(__name__)


def _requested_mode_or_none(tone: str) -> str | None:
    t = tone.strip().lower()
    aliases = {
        "honest": "honest",
        "exaggerated": "exaggerated",
        "absurd": "absurd",
        "hypothetical": "hypothetical",
    }
    return aliases.get(t)


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

    q = f"{request.situation} {request.tone} {request.target}"
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

    risk = analyze_risks(client, settings, drafts=drafts, situation=request.situation)
    logger.info("[%s] risk: analyzed %d modes", rid, len(risk))

    judge_scores = judge_drafts(client, settings, drafts=drafts, situation=request.situation)
    logger.info("[%s] judge: scores for %d modes", rid, len(judge_scores))

    best_mode, composite = select_best_draft(drafts, judge_scores, risk)
    requested_mode = _requested_mode_or_none(request.tone)
    if requested_mode:
        constrained = [d for d in drafts if d.mode.value == requested_mode]
        if constrained:
            constrained_mode, constrained_score = select_best_draft(constrained, judge_scores, risk)
            if constrained_mode is not None:
                logger.info(
                    "[%s] selection: enforcing requested tone=%s (best_mode=%s)",
                    rid,
                    requested_mode,
                    constrained_mode.value,
                )
                best_mode = constrained_mode
                composite = constrained_score
        else:
            logger.warning(
                "[%s] requested tone=%s not produced by generator; using global best",
                rid,
                requested_mode,
            )
    best_msg = None
    if best_mode is not None:
        for d in drafts:
            if d.mode == best_mode:
                best_msg = d.text
                break

    warnings: list[str] = []
    for d in drafts:
        r = risk.get(d.mode.value)
        if r:
            warnings.extend(r.warnings)
            if not r.framing_ok:
                warnings.append(f"Framing may be unclear for mode={d.mode.value}.")

    evidence_plan: list[EvidencePlanItem] = []
    evidence: list[EvidenceArtifact] = []
    if request.generate_evidence and best_mode is not None:
        best_draft = next(d for d in drafts if d.mode == best_mode)
        evidence_plan, evidence = generate_evidence_bundle(
            client, settings, request=request, best=best_draft
        )
        logger.info("[%s] evidence: %d artifacts", rid, len(evidence))

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
        evidence_plan=evidence_plan,
        evidence=evidence,
        meta={"model": settings.openai_model},
    )
