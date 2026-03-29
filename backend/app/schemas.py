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
    PROFESSIONAL = "professional"
    EMOTIONAL = "emotional"


# One draft per mode; fixed order for API stability.
DRAFT_MODES_ORDER: tuple[DraftMode, ...] = (
    DraftMode.HONEST,
    DraftMode.EXAGGERATED,
    DraftMode.ABSURD,
    DraftMode.PROFESSIONAL,
    DraftMode.EMOTIONAL,
)


class TargetAudience(StrEnum):
    """Who the fictional message is addressed to (controls register and structure)."""

    FRIEND = "friend"
    FAMILY = "family"
    COWORKER = "coworker"
    MANAGER = "manager"
    CLIENT = "client"
    PARTNER = "partner"
    TEACHER_PROFESSOR = "teacher_professor"
    FORMAL_OFFICIAL = "formal_official"


class GenerateRequest(BaseModel):
    """User input for narrative training simulation."""

    situation: str = Field(..., min_length=1, description="Scenario context (training only).")
    tone: DraftMode = Field(
        ...,
        description="Selected tone; matching draft is best_message (not judge-selected).",
    )
    target: TargetAudience = Field(
        ...,
        description="Recipient; shapes vocabulary, formality, and message length.",
    )
    existing_message: str | None = Field(
        default=None,
        description="Optional draft to rewrite as a fictional training example.",
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

    kind: str = Field(..., description='Planned artifact kind; use "generated_image" for HF image.')
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


class DraftImageRequest(BaseModel):
    """On-demand synthetic illustration for one draft (POST /api/evidence/image)."""

    situation: str = Field(..., min_length=1)
    target: TargetAudience
    draft_mode: DraftMode = Field(
        ...,
        description="Mode of the draft this image illustrates.",
    )
    draft_text: str = Field(..., min_length=1)


class DraftImageResponse(BaseModel):
    """Single synthetic image for a draft."""

    synthetic: bool = True
    non_verifiable: bool = True
    kind: str = "generated_image"
    caption: str
    image_base64: str
    mime_type: str = "image/png"


class GenerateResponse(BaseModel):
    """API response: best draft, scores, warnings."""

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
        description="Weighted composite for the selected tone's draft only (informational).",
    )
    warnings: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
