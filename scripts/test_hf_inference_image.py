#!/usr/bin/env python3
"""
Standalone Hugging Face Inference Providers smoke test for image generation.

Uses ``huggingface_hub.InferenceClient`` (router), not the deprecated
``api-inference.huggingface.co`` URL.

Usage (from repo root, with venv activated):
  python scripts/test_hf_inference_image.py

Optional:
  python scripts/test_hf_inference_image.py --model black-forest-labs/FLUX.1-schnell
  python scripts/test_hf_inference_image.py --prompt "a photo of a red car on a street"

Reads HUGGINGFACE_API_TOKEN from environment or from `.env` at repo root (same as the app).

If this script fails, the app will also fall back to PIL placeholders.
Fix token, model access, or network first.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

# Repo root = parent of scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv_simple(path: Path) -> None:
    """Set os.environ from KEY=VALUE lines (no export of bash syntax)."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def main() -> int:
    parser = argparse.ArgumentParser(description="Test HF image inference (InferenceClient)")
    parser.add_argument(
        "--model",
        default=os.environ.get("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"),
        help="Model id on Hugging Face Hub",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Photorealistic city street at night with heavy traffic, brake lights, "
            "cinematic lighting, no text"
        ),
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "scripts" / "hf_test_output.png"),
        help="Where to save the image (PNG)",
    )
    args = parser.parse_args()

    load_dotenv_simple(REPO_ROOT / ".env")
    token = os.environ.get("HUGGINGFACE_API_TOKEN", "").strip()
    if not token:
        print("ERROR: HUGGINGFACE_API_TOKEN is empty or missing.", file=sys.stderr)
        print("  Add it to .env at repo root or export it:", file=sys.stderr)
        print('  export HUGGINGFACE_API_TOKEN="hf_..."', file=sys.stderr)
        return 1

    masked = token[:4] + "..." + token[-4:] if len(token) >= 8 else "(short)"
    print(f"Token: loaded (length={len(token)}, masked={masked})")
    print(f"InferenceClient(provider=auto) model={args.model}")
    print(f"Prompt: {args.prompt[:120]}...")

    client = InferenceClient(api_key=token, provider="auto", timeout=120.0)
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            image = client.text_to_image(args.prompt, model=args.model)
        except HfHubHTTPError as e:
            status = e.response.status_code if e.response is not None else None
            print(f"Attempt {attempt}: HfHubHTTPError status={status} {str(e)[:400]}")
            if status == 503 and attempt < max_attempts:
                wait = 3.0
                if e.response is not None:
                    try:
                        body = e.response.json()
                        if isinstance(body, dict) and body.get("estimated_time") is not None:
                            wait = max(1.0, min(float(body["estimated_time"]), 20.0))
                    except ValueError:
                        pass
                print(f"Model loading (503), sleeping {wait:.1f}s...")
                time.sleep(wait)
                continue
            return 1
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data = buf.getvalue()
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        print(f"SUCCESS: wrote PNG ({len(data)} bytes) to {out_path}")
        print("Open this file to confirm the image looks correct.")
        return 0

    print("ERROR: exhausted retries", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
