#!/usr/bin/env python3
"""
Standalone Hugging Face Inference API smoke test for image generation.

Usage (from repo root, with venv activated):
  python scripts/test_hf_inference_image.py

Optional:
  python scripts/test_hf_inference_image.py --model runwayml/stable-diffusion-v1-5
  python scripts/test_hf_inference_image.py --prompt "a photo of a red car on a street"

Reads HUGGINGFACE_API_TOKEN from environment or from `.env` at repo root (same as the app).

If this script fails, the app will also fall back to PIL placeholders.
Fix token, model access, or network first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

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


def sniff_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    return "application/octet-stream"


def main() -> int:
    parser = argparse.ArgumentParser(description="Test HF image inference API")
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
        help="Where to save the image (PNG or JPEG)",
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

    url = f"https://api-inference.huggingface.co/models/{args.model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": args.prompt}

    print(f"POST {url}")
    print(f"Prompt: {args.prompt[:120]}...")

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            r = httpx.post(url, headers=headers, json=payload, timeout=120.0)
        except httpx.HTTPError as e:
            print(f"HTTP transport error: {e}", file=sys.stderr)
            return 1

        ctype = r.headers.get("content-type", "")
        print(
            f"Attempt {attempt}: status={r.status_code} "
            f"content-type={ctype} len={len(r.content)}"
        )

        if r.status_code == 200:
            if "application/json" in ctype:
                # Sometimes JSON wraps base64 or error
                try:
                    data = r.json()
                except json.JSONDecodeError:
                    print("ERROR: 200 but JSON parse failed", file=sys.stderr)
                    print(r.text[:800], file=sys.stderr)
                    return 1
                print("JSON body (truncated):", json.dumps(data, indent=2)[:1200])
                print(
                    "ERROR: Got JSON instead of raw image bytes. "
                    "Check model supports inference API / your token has access.",
                    file=sys.stderr,
                )
                return 1

            mime = sniff_mime(r.content)
            out_path = Path(args.out)
            if mime == "image/jpeg" and not str(out_path).lower().endswith((".jpg", ".jpeg")):
                out_path = out_path.with_suffix(".jpg")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(r.content)
            print(f"SUCCESS: wrote {mime} to {out_path}")
            print("Open this file to confirm the image looks correct.")
            return 0

        # Non-200: print body for debugging
        body_preview = r.text[:2000]
        print(f"Response body (truncated):\n{body_preview}")

        if r.status_code == 503:
            try:
                j = r.json()
                est = j.get("estimated_time") if isinstance(j, dict) else None
            except ValueError:
                est = None
            wait = 3.0
            if est is not None:
                wait = max(1.0, min(float(est), 20.0))
            print(f"Model loading (503), sleeping {wait:.1f}s...")
            if attempt < max_attempts:
                time.sleep(wait)
                continue
        return 1

    print("ERROR: exhausted retries", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
