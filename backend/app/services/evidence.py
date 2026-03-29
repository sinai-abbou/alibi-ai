"""Synthetic illustrative evidence: optional HF-generated image only."""

from __future__ import annotations

import base64

from PIL import Image, ImageDraw, ImageFont

from app.schemas import (
    DraftMode,
    EvidenceArtifact,
    EvidencePlanItem,
    GenerateRequest,
    MessageDraft,
    TargetAudience,
)
from app.utils.hf_image import generate_image_bytes, to_png_bytes_if_needed
from app.utils.logging import get_logger
from app.utils.settings import Settings

logger = get_logger(__name__)

# CLIP/SDXL heavily weight the *start* of the prompt (~77 tokens); long intros get truncated
# and the model never "sees" color instructions buried at the end. Keep color + subject up front.
_COLOR_SUBJECT_FIRST = (
    "Vibrant full-color realistic photograph, rgb, natural saturation, accurate white balance, "
    "blue sky and colored objects visible, not monochrome. "
)

_SINGLE_FRAME_SHORT = (
    "One photographic frame only, not comic panels, manga, speech bubbles, or split layout. "
)


def _prompt_header(*, subject_line: str) -> str:
    """Dense opening: color + medium + main subject (must survive token truncation)."""
    sub = subject_line.strip()
    if len(sub) > 420:
        sub = sub[:417] + "…"
    return f"{_COLOR_SUBJECT_FIRST}{_SINGLE_FRAME_SHORT}MAIN SUBJECT: {sub}\n"

_NEGATIVE_BAN_COMICS = (
    "comic strip, bande dessinée, bd, manga, graphic novel, sequential art, storyboard, "
    "multiple panels, panel borders, gutters, grid layout, split screen, collage layout, "
    "speech bubble, thought bubble, word balloon, dialogue balloon, caption box, "
    "cartoon style, comic book art, ligne claire, halftone, comic layout, webcomic, "
    "newspaper comic, anime style, cel shading, infographic panel"
)

# Listed first in the final negative string — if the API truncates, these matter most for color.
_NEGATIVE_BW_FIRST = (
    "black and white, monochrome, grayscale, greyscale, desaturated, duotone, sepia, film noir, "
    "pencil sketch, ink drawing, engraving, line art, storybook illustration"
)

_NEGATIVE_BAN_BW_SKETCH = (
    "b&w photo, ansel adams style, woodcut, lithograph, charcoal drawing, concept art sketch, "
    "color drained to gray, low saturation monochrome"
)

# Models often add unrelated stock "tiny people" (bikes, wheelchairs) — discourage unless in text.
_NEGATIVE_RANDOM_FIGURES = (
    "random pedestrian, stock photo crowd, tiny person, silhouette mascot, "
    "unrelated cyclist, bicycle rider, person on bike, wheelchair, wheelchair user, mobility scooter, "
    "gurney, hospital stretcher as focal subject, crutches as main subject, "
    "accessibility pictogram, disability symbol icon, isa wheelchair symbol, "
    "random bystander, walking figure, cartoon character person"
)


def plan_generated_image_only(
    *,
    request: GenerateRequest,
    best: MessageDraft,
) -> list[EvidencePlanItem]:
    """Single evidence item: HF image, caption derived from situation + mode (no LLM)."""
    caption = _evidence_image_caption(request, best)
    return [EvidencePlanItem(kind="generated_image", description=caption)]


def _evidence_image_caption(request: GenerateRequest, best: MessageDraft) -> str:
    """Short UI caption: absurd = scenario from message; else grounded delay theme."""
    situation = request.situation.strip()
    situation_short = situation[:160] + ("…" if len(situation) > 160 else "")
    if best.mode == DraftMode.ABSURD:
        raw = best.text.strip()
        excerpt = raw[:180] + ("…" if len(raw) > 180 else "")
        return f"Absurd scenario (photorealistic style): {excerpt}"
    scene, _ = _match_scenario(situation, best.text)
    if scene:
        first = scene.split(".")[0].strip()
        return f"Simulated photo (training): {first} — context: {situation_short}"
    return f"Simulated photo of the situation (training): {situation_short}"


# (keyword substrings) -> concrete scene line, extra negative-prompt terms
_SCENARIO_PATTERNS: list[tuple[tuple[str, ...], tuple[str, str]]] = [
    (
        (
            "traffic",
            "jam",
            "congestion",
            "gridlock",
            "highway",
            "bottleneck",
            "stuck in",
            "heavy traffic",
            "road",
            "commute",
            "rush hour",
        ),
        (
            "Dense urban traffic at rush hour: multi-lane road or highway, "
            "long lines of vehicles, red brake lights, wet or dry asphalt, "
            "slight haze, elevated or street-level view, "
            "natural daylight with realistic colors",
            (
                "office interior, conference room, meeting table, "
                "job interview room, handshake, infographic"
            ),
        ),
    ),
    (
        (
            "train",
            "subway",
            "metro",
            "transit",
            "missed the",
            "platform",
            "cancellation",
        ),
        (
            "Urban train or subway station: platform, digital delay boards or departure screens, "
            "tracks or tunnel mouth, station architecture, realistic ambient light; "
            "environment and signage in focus, not a crowd portrait",
            "empty office, corporate boardroom, infographic, highway traffic jam",
        ),
    ),
    (
        ("rain", "storm", "snow", "weather", "flood", "wind"),
        (
            "Bad weather outdoors: heavy rain or snow on streets, umbrellas, reduced visibility, "
            "realistic meteorological conditions affecting travel",
            "sunny office, meeting room, infographic, clear sky highway without weather",
        ),
    ),
    (
        ("alarm", "overslept", "woke up", "oversleep"),
        (
            "Early morning bedroom: bed, alarm clock glowing, soft dawn light through window, "
            "sense of rushing morning",
            "office meeting, highway traffic, infographic",
        ),
    ),
    (
        ("tire", "flat tire", "breakdown", "car trouble", "engine"),
        (
            "Roadside or parking area: car with hazard lights, open hood or flat tire, "
            "realistic automotive detail",
            "office interior, conference room, infographic",
        ),
    ),
    (
        ("sick", "hospital", "child", "family emergency", "caregiv"),
        (
            "Home or care context suggesting an urgent family situation (no identifiable faces), "
            "warm indoor color photograph",
            "corporate meeting, traffic jam infographic, handshake",
        ),
    ),
]


def _match_scenario(situation: str, message: str) -> tuple[str | None, str | None]:
    blob = f"{situation} {message}".lower()
    for keywords, (scene, negative_extra) in _SCENARIO_PATTERNS:
        if any(k in blob for k in keywords):
            return scene, negative_extra
    return None, None


def _planner_is_generic_advice(description: str) -> bool:
    d = description.lower()
    return any(
        x in d
        for x in (
            "infographic",
            "tips for",
            "handling late",
            "accountability",
            "workplace tips",
            "advice graphic",
            "checklist",
        )
    )


def _build_absurd_image_prompt_pair(
    request: GenerateRequest,
    best: MessageDraft,
    planner_description: str,
) -> tuple[str, str | None]:
    """Photorealistic render of the literal absurd scenario (surreal content allowed)."""
    situation = request.situation.strip()[:400]
    best_text = best.text.strip()[:500]
    tone = request.tone.value[:60]
    target = request.target.value[:60]
    hint = planner_description.strip()[:400]

    # Put the absurd scenario in the header so CLIP sees color + subject before truncation.
    header = _prompt_header(subject_line=best_text or situation)
    lead = (
        header
        + "Training simulation only, not real evidence. ABSURD MODE: depict this scenario "
        "literally, surreal elements allowed, photorealistic color 3d look, natural light, "
        "sharp textures, depth of field.\n"
        f"Situation context: {situation}\n"
        f"Tone: {tone}; Audience: {target}\n"
    )
    if hint and not _planner_is_generic_advice(hint):
        lead += f"Nuance: {hint}\n"

    safety = (
        "No readable text, logos, watermarks, comic panels, or bubbles. "
        "Keep rich natural color throughout. "
        "No random crowds, cyclists, wheelchairs, or accessibility icons unless the scenario says so."
    )
    positive = f"{lead}{safety}"

    negative = (
        f"{_NEGATIVE_BW_FIRST}, {_NEGATIVE_BAN_BW_SKETCH}, {_NEGATIVE_BAN_COMICS}, "
        f"{_NEGATIVE_RANDOM_FIGURES}, watermark, logo, stick figure, flat 2d illustration, "
        "diagram, chart, bullet points"
    )
    return positive, negative


def _build_grounded_image_prompt_pair(
    request: GenerateRequest,
    best: MessageDraft,
    planner_description: str,
) -> tuple[str, str | None]:
    """Real-world delay causes; no surreal metaphors driving the scene."""
    situation = request.situation.strip()[:400]
    best_text = best.text.strip()[:500]
    tone = request.tone.value[:60]
    target = request.target.value[:60]
    hint = planner_description.strip()[:400]

    scene, neg_extra = _match_scenario(situation, best_text)

    # Short anchor in the header (early tokens); full scene line after if matched.
    if scene:
        anchor = scene.split(".")[0].strip() or scene
        header = _prompt_header(subject_line=anchor)
    else:
        header = _prompt_header(subject_line=f"{situation} — {best_text}")

    lead = (
        header
        + "Training simulation only. Depict a plausible real-world lateness "
        "(commute, weather, transit, health, vehicle). One photo, not comic layout.\n"
    )
    if scene:
        lead += f"Scene detail: {scene}\n"
    else:
        lead += (
            "Show the specific delay circumstance (transport, weather, health, breakdown) "
            "from the situation; avoid generic office stock imagery.\n"
        )

    block = (
        f"Situation: {situation}\n"
        f"Message: {best_text}\n"
        f"Tone: {tone}; Audience: {target}\n"
    )
    if hint and not _planner_is_generic_advice(hint):
        block += f"Nuance (realistic): {hint}\n"
    elif hint and _planner_is_generic_advice(hint):
        block += "Ignore infographic-style planner hints; use literal delay from text above.\n"

    safety = (
        "No readable text, logos, watermarks, bubbles, forged documents, UI overlays. "
        "Rich natural color, outdoor or indoor hues true to life. "
        "No random pedestrians, cyclists, wheelchairs, or pictograms unless the message says so; "
        "emphasize roads, vehicles, weather, buildings."
    )
    positive = f"{lead}{block}{safety}"

    negative_core = (
        f"{_NEGATIVE_BW_FIRST}, {_NEGATIVE_BAN_BW_SKETCH}, {_NEGATIVE_BAN_COMICS}, "
        f"{_NEGATIVE_RANDOM_FIGURES}, infographic, diagram, chart, bullet points, "
        "corporate poster, stock handshake, empty conference room, whiteboard, meeting table, "
        "interview room, tips list, watermark, logo"
    )
    negative = f"{negative_core}, {neg_extra}" if neg_extra else negative_core

    return positive, negative


def _build_image_prompt_pair(
    request: GenerateRequest,
    best: MessageDraft,
    planner_description: str = "",
) -> tuple[str, str | None]:
    """Return (positive, negative). Absurd mode: literal surreal scenario, photorealistic."""
    if best.mode == DraftMode.ABSURD:
        return _build_absurd_image_prompt_pair(request, best, planner_description)
    return _build_grounded_image_prompt_pair(request, best, planner_description)


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


def build_evidence_artifacts(
    settings: Settings,
    *,
    request: GenerateRequest,
    best: MessageDraft,
    plan: list[EvidencePlanItem],
) -> list[EvidenceArtifact]:
    artifacts: list[EvidenceArtifact] = []
    for item in plan:
        if item.kind != "generated_image":
            logger.warning("Evidence: skipping non-generated_image plan item %s", item.kind)
            continue
        prompt, negative = _build_image_prompt_pair(request, best, item.description)
        data = generate_image_bytes(settings, prompt, negative_prompt=negative)
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
            png = _pil_mock_screenshot("HF image unavailable; PIL fallback", item.description[:80])
            artifacts.append(
                EvidenceArtifact(
                    synthetic=True,
                    non_verifiable=True,
                    kind="generated_image",
                    caption=item.description + " (fallback image)",
                    image_base64=base64.b64encode(png).decode("ascii"),
                    mime_type="image/png",
                )
            )
    return artifacts


def generate_evidence_bundle(
    settings: Settings,
    *,
    request: GenerateRequest,
    best: MessageDraft,
) -> tuple[list[EvidencePlanItem], list[EvidenceArtifact]]:
    logger.info("Evidence: single synthetic image (no mock chat/screenshot)")
    plan = plan_generated_image_only(request=request, best=best)
    arts = build_evidence_artifacts(settings, request=request, best=best, plan=plan)
    return plan, arts


def generate_draft_illustration(
    settings: Settings,
    *,
    situation: str,
    target: TargetAudience,
    draft_mode: DraftMode,
    draft_text: str,
) -> EvidenceArtifact:
    """Build one synthetic image for a specific draft (used by POST /api/evidence/image)."""
    request = GenerateRequest(
        situation=situation,
        tone=draft_mode,
        target=target,
        existing_message=None,
    )
    best = MessageDraft(mode=draft_mode, text=draft_text)
    _plan, arts = generate_evidence_bundle(settings, request=request, best=best)
    if not arts:
        raise RuntimeError("No illustration artifact produced")
    return arts[0]
