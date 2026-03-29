"""Optional Hugging Face image generation via Inference Providers (router)."""

from __future__ import annotations

import io
import time

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

from app.utils.logging import get_logger
from app.utils.settings import Settings

logger = get_logger(__name__)


def generate_image_bytes(
    settings: Settings,
    prompt: str,
    *,
    negative_prompt: str | None = None,
    timeout_s: float = 120.0,
) -> bytes | None:
    """Return PNG bytes, or None on failure/missing token.

    Uses ``InferenceClient`` (router.huggingface.co), not the deprecated
    ``api-inference.huggingface.co`` endpoint.
    """
    token = settings.huggingface_api_token
    if not token:
        logger.info("HF image: no HUGGINGFACE_API_TOKEN; skipping API call")
        return None
    logger.info("HF image: token configured from env (.env) = yes")
    model = settings.hf_image_model
    logger.info(
        "HF image: prompt_len=%s neg_len=%s model=%s",
        len(prompt),
        len(negative_prompt) if negative_prompt else 0,
        model,
    )
    client = InferenceClient(api_key=token, provider="auto", timeout=timeout_s)
    max_attempts = 3

    def _t2i(extra: dict[str, float | int | str]) -> object:
        t2i_kw: dict[str, float | int | str] = dict(extra)
        if negative_prompt:
            t2i_kw["negative_prompt"] = negative_prompt
        return client.text_to_image(prompt, model=model, **t2i_kw)

    rich_params: dict[str, float | int | str] = {
        "guidance_scale": settings.hf_image_guidance_scale,
        "num_inference_steps": settings.hf_image_num_inference_steps,
        "width": settings.hf_image_width,
        "height": settings.hf_image_height,
    }

    for attempt in range(1, max_attempts + 1):
        try:
            try:
                image = _t2i(rich_params)
            except TypeError:
                try:
                    image = _t2i({"guidance_scale": settings.hf_image_guidance_scale})
                except TypeError:
                    image = _t2i({})
            except HfHubHTTPError as e2:
                status2 = e2.response.status_code if e2.response is not None else None
                if status2 in (400, 422):
                    logger.info(
                        "HF image: retrying without resolution/steps (status=%s)", status2
                    )
                    image = _t2i({"guidance_scale": settings.hf_image_guidance_scale})
                else:
                    raise
        except HfHubHTTPError as e:
            status = e.response.status_code if e.response is not None else None
            logger.warning(
                "HF image: attempt=%s HfHubHTTPError status=%s message=%s",
                attempt,
                status,
                str(e)[:500],
            )
            if status == 503 and attempt < max_attempts:
                wait_s = 3.0
                if e.response is not None:
                    try:
                        body = e.response.json()
                        if isinstance(body, dict) and body.get("estimated_time") is not None:
                            wait_s = max(1.0, min(float(body["estimated_time"]), 20.0))
                    except ValueError:
                        pass
                time.sleep(wait_s)
                continue
            return None
        except OSError as e:
            logger.warning("HF image: attempt=%s failed: %s", attempt, e)
            return None
        except Exception as e:  # noqa: BLE001
            # Provider may raise beyond HfHubHTTPError; never fail the pipeline.
            logger.warning("HF image: attempt=%s unexpected error: %s", attempt, e)
            return None

        # Normalize to RGB for PNG; log if the API returned true grayscale (debugging color issues).
        if hasattr(image, "mode"):
            if image.mode == "L":
                logger.warning(
                    "HF image: provider returned mode=L (grayscale); prompt may be truncated or "
                    "model/router ignored color — check HF_IMAGE_MODEL and server logs."
                )
            image = image.convert("RGB")

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data = buf.getvalue()
        logger.info(
            "HF image: success attempt=%s model=%s png_bytes=%s",
            attempt,
            model,
            len(data),
        )
        return data

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
