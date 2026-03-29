"""Synthetic illustrative evidence: mock UI cards and optional HF image."""

from __future__ import annotations

import base64
import json

from PIL import Image, ImageDraw, ImageFont

from app.schemas import EvidenceArtifact, EvidencePlanItem, GenerateRequest, MessageDraft
from app.utils.hf_image import generate_image_bytes, to_png_bytes_if_needed
from app.utils.logging import get_logger
from app.utils.openai_client import OpenAIClient
from app.utils.settings import Settings

logger = get_logger(__name__)

SYSTEM = """You plan synthetic illustrative evidence for a training UI. Output JSON only.
Evidence is never real proof. Use kinds like mock_chat, mock_screenshot, generated_image."""


def plan_evidence(
    client: OpenAIClient,
    settings: Settings,
    *,
    request: GenerateRequest,
    best: MessageDraft,
) -> list[EvidencePlanItem]:
    user = json.dumps(
        {
            "situation": request.situation,
            "tone": request.tone,
            "target": request.target,
            "best_mode": best.mode.value,
            "best_text": best.text,
        }
    )
    user_prompt = (
        'Create 3 plan items. Return JSON: { "plan": [ { "kind": "...", '
        '"description": "..." } ] }. You MUST include one item with kind="generated_image".\n'
        + user
    )
    raw = client.chat_json(system=SYSTEM, user=user_prompt, temperature=0.4)
    items: list[EvidencePlanItem] = []
    for row in raw.get("plan", []):
        if not isinstance(row, dict):
            continue
        kind = str(row.get("kind", "mock_screenshot"))
        desc = str(row.get("description", ""))
        items.append(EvidencePlanItem(kind=kind, description=desc))
    if not any(it.kind == "generated_image" for it in items):
        items.insert(
            0,
            EvidencePlanItem(
                kind="generated_image",
                description="Realistic illustration of the situation context (simulated)",
            ),
        )
    if not items:
        items = [
            EvidencePlanItem(
                kind="generated_image",
                description="Realistic illustration of the situation context (simulated)",
            ),
            EvidencePlanItem(
                kind="mock_chat",
                description="Synthetic chat snippet for UX demo",
            ),
            EvidencePlanItem(
                kind="mock_screenshot",
                description="Synthetic notification-style card",
            ),
        ]
    return items


def _situation_image_prompt(request: GenerateRequest, best: MessageDraft) -> str:
    """Build a rich text-to-image prompt for a realistic simulated illustration."""
    situation = request.situation.strip()[:300]
    tone = request.tone.strip()[:60]
    target = request.target.strip()[:60]
    best_text = best.text.strip()[:240]
    return (
        "Illustration of a situation for training UI, simulated context only, not evidence. "
        "Generate a realistic, photo-like scene with cinematic lighting, high detail, "
        "natural colors, and believable environment.\n"
        f"Situation context: {situation}\n"
        f"Message context: {best_text}\n"
        f"Tone: {tone}; Audience: {target}\n"
        "No logos, no readable personal data, no forged documents, no screenshot UI overlays."
    )


def _pil_mock_screenshot(caption: str, subtitle: str) -> bytes:
    w, h = 640, 360
    img = Image.new("RGB", (w, h), color=(245, 246, 250))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([20, 20, w - 20, h - 20], outline=(180, 186, 199), width=2)
    draw.text((40, 50), "Synthetic UI preview (training only)", fill=(40, 44, 52), font=font)
    draw.text((40, 90), subtitle[:80], fill=(80, 86, 96), font=font)
    draw.text((40, 140), caption[:120], fill=(30, 34, 42), font=font)
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _pil_mock_chat(lines: list[str]) -> bytes:
    w, h = 520, 420
    img = Image.new("RGB", (w, h), color=(230, 235, 240))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    y = 20
    for line in lines[:8]:
        draw.text((20, y), line[:70], fill=(20, 24, 32), font=font)
        y += 28
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_evidence_artifacts(
    settings: Settings,
    *,
    request: GenerateRequest,
    best: MessageDraft,
    plan: list[EvidencePlanItem],
) -> list[EvidenceArtifact]:
    artifacts: list[EvidenceArtifact] = []
    for item in plan:
        if item.kind == "generated_image":
            prompt = _situation_image_prompt(request, best)
            data = generate_image_bytes(settings, prompt)
            if data:
                png, mime = to_png_bytes_if_needed(data)
                b64 = base64.b64encode(png).decode("ascii")
                artifacts.append(
                    EvidenceArtifact(
                        synthetic=True,
                        non_verifiable=True,
                        kind="generated_image",
                        caption=f"{item.description} (simulated context, not evidence)",
                        image_base64=b64,
                        mime_type=mime,
                    )
                )
            else:
                png = _pil_mock_screenshot("HF image unavailable; PIL fallback", item.description)
                artifacts.append(
                    EvidenceArtifact(
                        synthetic=True,
                        non_verifiable=True,
                        kind="mock_screenshot",
                        caption=item.description + " (fallback image)",
                        image_base64=base64.b64encode(png).decode("ascii"),
                        mime_type="image/png",
                    )
                )
        elif item.kind == "mock_chat":
            lines = [
                "[Synthetic thread — training only]",
                f"To: {request.target}",
                f"Mode: {best.mode.value}",
                "",
                best.text[:280],
            ]
            png = _pil_mock_chat(lines)
            artifacts.append(
                EvidenceArtifact(
                    synthetic=True,
                    non_verifiable=True,
                    kind="mock_chat",
                    caption=item.description,
                    text_content="\n".join(lines),
                    image_base64=base64.b64encode(png).decode("ascii"),
                    mime_type="image/png",
                )
            )
        else:
            png = _pil_mock_screenshot(best.text[:120], item.description)
            artifacts.append(
                EvidenceArtifact(
                    synthetic=True,
                    non_verifiable=True,
                    kind="mock_screenshot",
                    caption=item.description,
                    image_base64=base64.b64encode(png).decode("ascii"),
                    mime_type="image/png",
                )
            )
    return artifacts


def generate_evidence_bundle(
    client: OpenAIClient,
    settings: Settings,
    *,
    request: GenerateRequest,
    best: MessageDraft,
) -> tuple[list[EvidencePlanItem], list[EvidenceArtifact]]:
    logger.info("Evidence: planning synthetic artifacts")
    plan = plan_evidence(client, settings, request=request, best=best)
    arts = build_evidence_artifacts(settings, request=request, best=best, plan=plan)
    return plan, arts
