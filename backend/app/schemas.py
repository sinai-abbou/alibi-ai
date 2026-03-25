"""Pydantic models for API requests/responses and pipeline payloads."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DraftMode(StrEnum):
    """Training/simulation style of the fictional example message."""

    HONEST = "honest"
    EXAGGERATED = "exaggerated"
    ABSURD = "absurd"
    HYPOTHETICAL = "hypothetical"


class GenerateRequest(BaseModel):
    """User input for narrative training simulation."""

    situation: str = Field(..., min_length=1, description="Scenario context (training only).")
    tone: str = Field(..., min_length=1, description="Desired tone, e.g. formal, casual.")
    target: str = Field(..., min_length=1, description="Audience, e.g. manager, friend.")
    existing_message: str | None = Field(
        default=None,
        description="Optional draft to rewrite as a fictional training example.",
    )
    generate_evidence: bool = Field(
        default=False,
        description="If true, include synthetic illustrative evidence (non-verifiable).",
    )


class MessageDraft(BaseModel):
    """A single fictional training message draft."""

    mode: DraftMode
    text: str = Field(..., min_length=1)
    fictional_framing_present: bool = Field(
        default=True,
        description="Whether the draft explicitly frames content as fictional/training.",
    )


class JudgeScores(BaseModel):
    """Per-draft scores from the Judge agent (0–10 scale)."""

    plausibility: float = Field(..., ge=0, le=10)
    coherence: float = Field(..., ge=0, le=10)
    training_compliance: float = Field(..., ge=0, le=10)


class RiskPerDraft(BaseModel):
    """Risk analyzer output for one draft."""

    policy_risk: float = Field(..., ge=0, le=10, description="Higher = more concerning.")
    warnings: list[str] = Field(default_factory=list)
    framing_ok: bool = True


class EvidencePlanItem(BaseModel):
    """Planned synthetic evidence artifact."""

    kind: str = Field(..., description="e.g. mock_chat, mock_screenshot, generated_image")
    description: str


class EvidenceArtifact(BaseModel):
    """Synthetic illustrative evidence (never real proof)."""

    synthetic: bool = True
    non_verifiable: bool = True
    kind: str
    caption: str
    text_content: str | None = None
    image_base64: str | None = Field(
        default=None,
        description="PNG/JPEG as base64 for UI display.",
    )
    mime_type: str | None = "image/png"


class GenerateResponse(BaseModel):
    """API response: best draft, scores, warnings, optional evidence."""

    request_id: str
    situation_summary: str
    rag_context_used: list[str] = Field(default_factory=list)
    drafts: list[MessageDraft]
    judge_scores: dict[str, JudgeScores] = Field(
        default_factory=dict,
        description="Scores keyed by draft mode value.",
    )
    risk: dict[str, RiskPerDraft] = Field(default_factory=dict)
    best_mode: DraftMode | None = None
    best_message: str | None = None
    composite_score: float | None = Field(
        default=None,
        description="Weighted composite used for selection (higher is better).",
    )
    warnings: list[str] = Field(default_factory=list)
    evidence_plan: list[EvidencePlanItem] = Field(default_factory=list)
    evidence: list[EvidenceArtifact] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
