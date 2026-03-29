"""Tests for evidence image prompt construction (scene alignment with excuses)."""

from __future__ import annotations

from app.schemas import DraftMode, GenerateRequest, MessageDraft
from app.services.evidence import (
    _build_image_prompt_pair,
    _match_scenario,
    _planner_is_generic_advice,
)


def test_match_scenario_traffic_from_message() -> None:
    scene, neg = _match_scenario(
        "late to interview",
        "Sorry, I hit unexpected traffic on the highway.",
    )
    assert scene is not None
    assert "traffic" in scene.lower() or "vehicle" in scene.lower() or "rush" in scene.lower()
    assert neg is not None
    assert "conference" in neg.lower() or "office" in neg.lower()


def test_match_scenario_none_for_vague() -> None:
    scene, neg = _match_scenario("x", "I am sorry I was late.")
    assert scene is None
    assert neg is None


def test_planner_generic_advice() -> None:
    assert _planner_is_generic_advice("An infographic with tips for handling late arrivals")
    assert not _planner_is_generic_advice("Night highway with brake lights and rain")


def test_build_prompt_stresses_cause_not_office() -> None:
    req = GenerateRequest(
        situation="I got 1 hour late to a job interview",
        tone="honest",
        target="manager",
        existing_message=None,
    )
    best = MessageDraft(
        mode=DraftMode.HONEST,
        text=(
            "I apologize for arriving late. I encountered unexpected traffic "
            "that delayed my arrival."
        ),
    )
    pos, neg = _build_image_prompt_pair(
        req,
        best,
        "Infographic about professional punctuality tips",
    )
    assert "Scene detail:" in pos or "traffic" in pos.lower()
    assert "traffic" in pos.lower()
    assert "infographic" in neg.lower()
    assert "conference" in neg.lower() or "meeting" in neg.lower()
    assert "monochrome" in neg.lower() or "black and white" in neg.lower()
    assert "vibrant full-color" in pos.lower() or "full-color" in pos.lower()
    assert "MAIN SUBJECT:" in pos


def test_build_prompt_absurd_follows_literal_scenario() -> None:
    req = GenerateRequest(
        situation="Late to a fancy dinner reservation",
        tone="absurd",
        target="friend",
        existing_message=None,
    )
    best = MessageDraft(
        mode=DraftMode.ABSURD,
        text=(
            "Sorry I'm late — the restaurant filled with dancing penguins in tuxedos "
            "blocked the door."
        ),
    )
    pos, neg = _build_image_prompt_pair(req, best, "")
    assert "ABSURD MODE" in pos
    assert "penguin" in pos.lower()
    assert "penguin" not in neg.lower()
    assert "speech bubble" in neg.lower()
    assert "MAIN SUBJECT:" in pos
