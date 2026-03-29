"""Optional Hugging Face Inference API image generation."""

from __future__ import annotations

import io
import time
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
    logger.info("HF image: token configured from env (.env) = yes")
    model = settings.hf_image_model
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload: dict[str, Any] = {"inputs": prompt}
    try:
        with httpx.Client(timeout=timeout_s) as client:
            # Hugging Face Inference can return 503 while loading model.
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                r = client.post(url, headers=headers, json=payload)
                ctype = r.headers.get("content-type", "")
                logger.info(
                    "HF image: attempt=%s status=%s content_type=%s bytes=%s",
                    attempt,
                    r.status_code,
                    ctype,
                    len(r.content),
                )
                if r.status_code == 200 and "application/json" not in ctype:
                    return r.content
                # Log exact response snippet for debugging.
                logger.warning(
                    "HF image: response status=%s headers=%s body=%s",
                    r.status_code,
                    dict(r.headers),
                    r.text[:500],
                )
                if r.status_code != 503:
                    return None
                wait_s = 2.0
                try:
                    body = r.json()
                    if isinstance(body, dict) and body.get("estimated_time") is not None:
                        wait_s = max(1.0, min(float(body["estimated_time"]), 12.0))
                except ValueError:
                    pass
                if attempt < max_attempts:
                    time.sleep(wait_s)
        return None
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
