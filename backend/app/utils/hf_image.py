"""Optional Hugging Face Inference API image generation."""

from __future__ import annotations

import io
from typing import Any

import httpx

from app.utils.logging import get_logger
from app.utils.settings import Settings

logger = get_logger(__name__)


def generate_image_bytes(
    settings: Settings,
    prompt: str,
    *,
    timeout_s: float = 120.0,
) -> bytes | None:
    """Return PNG/JPEG bytes, or None on failure/missing token."""
    token = settings.huggingface_api_token
    if not token:
        logger.info("HF image: no HUGGINGFACE_API_TOKEN; skipping API call")
        return None
    model = settings.hf_image_model
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload: dict[str, Any] = {"inputs": prompt}
    try:
        with httpx.Client(timeout=timeout_s) as client:
            r = client.post(url, headers=headers, json=payload)
        if r.status_code != 200:
            logger.warning("HF image: HTTP %s %s", r.status_code, r.text[:200])
            return None
        ctype = r.headers.get("content-type", "")
        if "application/json" in ctype:
            # error payload
            logger.warning("HF image: unexpected JSON response")
            return None
        return r.content
    except httpx.HTTPError as e:
        logger.warning("HF image: request failed: %s", e)
        return None


def sniff_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    return "application/octet-stream"


def to_png_bytes_if_needed(data: bytes) -> tuple[bytes, str]:
    """Ensure PNG for UI; convert JPEG via PIL if needed."""
    mime = sniff_mime(data)
    if mime == "image/png":
        return data, "image/png"
    try:
        from PIL import Image

        im = Image.open(io.BytesIO(data)).convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    except Exception:
        return data, mime
