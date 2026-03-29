"""Streamlit UI for Alibi AI (training/simulation)."""

from __future__ import annotations

import base64
import os

import httpx
import streamlit as st

DEFAULT_BASE = os.environ.get("ALIBI_API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Alibi AI — training lab", layout="wide")
st.title("Alibi AI")
st.caption(
    "Communication training and simulation. Outputs are fictional examples for learning — "
    "not real-world proof or deception."
)

with st.sidebar:
    st.header("API")
    base = st.text_input("API base URL", value=DEFAULT_BASE)
    st.caption("Set `ALIBI_API_BASE` env var to change the default.")

situation = st.text_area("Situation (training context)", height=120)
tone = st.text_input("Tone", value="neutral")
target = st.text_input("Target audience", value="peer")
existing = st.text_area("Optional existing message (rewrite as fictional example)", height=80)
gen_evidence = st.checkbox("Include synthetic illustrative evidence (non-verifiable)", value=False)

if st.button("Generate", type="primary"):
    payload = {
        "situation": situation or "(unspecified)",
        "tone": tone or "neutral",
        "target": target or "reader",
        "existing_message": existing.strip() or None,
        "generate_evidence": gen_evidence,
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(f"{base.rstrip('/')}/api/generate", json=payload)
    except httpx.HTTPError as e:
        st.error(f"Request failed: {e}")
        st.stop()
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
        st.stop()
    data = r.json()
    st.subheader("Best message (training selection)")
    st.write(data.get("best_message") or "—")
    st.metric("Composite score", f"{data.get('composite_score') or 0:.2f}")
    if data.get("warnings"):
        st.warning("\n".join(data["warnings"]))

    st.subheader("Drafts")
    for d in data.get("drafts", []):
        with st.expander(f"{d.get('mode', '?')}"):
            st.write(d.get("text", ""))

    js = data.get("judge_scores") or {}
    if js:
        st.subheader("Judge scores")
        st.json(js)

    ev = data.get("evidence") or []
    if ev:
        st.subheader("Simulated context illustrations (not evidence)")
        st.caption(
            "These visuals are synthetic illustrations generated for UX/training. "
            "They are non-verifiable and must not be treated as proof."
        )
        for art in ev:
            st.write(f"**{art.get('kind', 'artifact')}** — {art.get('caption', '')}")
            b64 = art.get("image_base64")
            if b64:
                raw = base64.b64decode(b64)
                st.image(raw, caption="Synthetic / non-verifiable")

    st.caption(f"Request ID: {data.get('request_id', '')}")
