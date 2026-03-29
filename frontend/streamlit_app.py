"""Streamlit UI for Alibi AI (training/simulation)."""

from __future__ import annotations

import base64
import html
import os
import time

import httpx
import streamlit as st

DEFAULT_BASE = os.environ.get("ALIBI_API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Alibi AI — training lab", layout="wide")

# Hide Streamlit's default top-right "Running" / status spinner; we use a centered overlay instead.
st.markdown(
    """
    <style>
    [data-testid="stStatusWidget"] { display: none !important; }
    div[data-testid="stToolbarStatus"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _fullpage_loading_markup(title: str, detail: str) -> str:
    safe_title = html.escape(title)
    safe_detail = html.escape(detail)
    return f"""
<style>
@keyframes alibi-spin {{ to {{ transform: rotate(360deg); }} }}
#alibi-loading-root {{
  position: fixed;
  inset: 0;
  z-index: 999999;
  background: rgba(250, 250, 252, 0.97);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  font-family: system-ui, -apple-system, sans-serif;
  color: #1f2937;
}}
#alibi-loading-root .alibi-wheel {{
  width: 52px;
  height: 52px;
  border: 5px solid #e5e7eb;
  border-top-color: #dc2626;
  border-radius: 50%;
  animation: alibi-spin 0.75s linear infinite;
}}
#alibi-loading-root h2 {{
  margin: 0;
  font-size: 1.35rem;
  font-weight: 600;
}}
#alibi-loading-root p {{
  margin: 0;
  max-width: 28rem;
  text-align: center;
  line-height: 1.45;
  color: #4b5563;
  font-size: 0.95rem;
}}
</style>
<div id="alibi-loading-root">
  <div class="alibi-wheel" aria-hidden="true"></div>
  <h2>{safe_title}</h2>
  <p>{safe_detail}</p>
</div>
"""


st.title("Alibi AI")
st.caption(
    "Communication training and simulation. Outputs are fictional examples for learning — "
    "not real-world proof or deception."
)

with st.sidebar:
    st.header("API")
    base = st.text_input("API base URL", value=DEFAULT_BASE)
    st.caption("Set `ALIBI_API_BASE` env var to change the default.")

# Illustration request: defer to next run so the full-page overlay renders at root (same UX as Generate).
_PENDING_ILLUSTRATION = "alibi_pending_illustration"
_EXPAND_DRAFT = "alibi_expand_draft"

pending_illu = st.session_state.pop(_PENDING_ILLUSTRATION, None)
if pending_illu is not None and st.session_state.get("generate_result"):
    overlay_ill = st.empty()
    overlay_ill.markdown(
        _fullpage_loading_markup(
            "Generating illustration…",
            "Please wait — this may take a few seconds. Avoid closing this tab while it runs.",
        ),
        unsafe_allow_html=True,
    )
    time.sleep(0.12)
    mode_pending = pending_illu["mode"]
    body_ill = pending_illu["body"]
    draft_images_early = st.session_state.setdefault("draft_images", {})
    ill_errors_early = st.session_state.setdefault("illustration_errors", {})
    try:
        with httpx.Client(timeout=120.0) as client:
            r_ill = client.post(f"{base.rstrip('/')}/api/evidence/image", json=body_ill)
    except httpx.HTTPError as e:
        overlay_ill.empty()
        ill_errors_early[mode_pending] = str(e)
        draft_images_early.pop(mode_pending, None)
    else:
        overlay_ill.empty()
        if r_ill.status_code == 200:
            ill_errors_early.pop(mode_pending, None)
            draft_images_early[mode_pending] = r_ill.json()
            st.session_state[_EXPAND_DRAFT] = mode_pending
            try:
                st.toast("Illustration ready.", icon="🖼️")
            except Exception:
                pass
        else:
            ill_errors_early[mode_pending] = r_ill.text
            draft_images_early.pop(mode_pending, None)

situation = st.text_area("Situation (training context)", height=120)

TONE_OPTIONS = ["honest", "exaggerated", "absurd", "professional", "emotional"]
tone = st.selectbox(
    "Tone",
    options=TONE_OPTIONS,
    index=0,
    help=(
        "The highlighted “best message” always matches this tone. Judge scores are for comparison "
        "only."
    ),
)

AUDIENCE_OPTIONS: dict[str, str] = {
    "friend": "Friend",
    "family": "Family",
    "coworker": "Coworker",
    "manager": "Manager",
    "client": "Client",
    "partner": "Partner (romantic)",
    "teacher_professor": "Teacher / professor",
    "formal_official": "Formal (generic official)",
}
target = st.selectbox(
    "Target audience",
    options=list(AUDIENCE_OPTIONS.keys()),
    index=list(AUDIENCE_OPTIONS.keys()).index("friend"),
    format_func=lambda k: AUDIENCE_OPTIONS[k],
    help="Recipient shapes vocabulary, formality, and structure of every draft.",
)
existing = st.text_area("Optional existing message (rewrite as fictional example)", height=80)

col_gen, col_hint = st.columns([1, 3])
with col_gen:
    generate_clicked = st.button("Generate", type="primary", use_container_width=True)
with col_hint:
    st.caption(
        "A **centered loading wheel** blocks the page while the request runs "
        "(may take a few seconds)."
    )

if generate_clicked:
    overlay = st.empty()
    overlay.markdown(
        _fullpage_loading_markup(
            "Generating drafts…",
            "Please wait — this may take a few seconds. Avoid closing this tab while it runs.",
        ),
        unsafe_allow_html=True,
    )
    time.sleep(0.12)
    payload = {
        "situation": situation or "(unspecified)",
        "tone": tone,
        "target": target,
        "existing_message": existing.strip() or None,
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(f"{base.rstrip('/')}/api/generate", json=payload)
    except httpx.HTTPError as e:
        overlay.empty()
        st.error(f"Request failed: {e}")
        st.stop()
    overlay.empty()
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
        st.stop()
    st.session_state["generate_result"] = r.json()
    st.session_state["draft_images"] = {}
    st.session_state["illustration_errors"] = {}
    st.session_state["form_situation"] = situation or "(unspecified)"
    st.session_state["form_target"] = target
    st.session_state.pop(_EXPAND_DRAFT, None)
    try:
        st.toast("Drafts ready.", icon="✅")
    except Exception:
        pass

data = st.session_state.get("generate_result")
if data:
    st.subheader("Best message (training selection)")
    st.caption("Matches your selected tone above — not chosen by the judge.")
    st.write(data.get("best_message") or "—")
    st.metric("Composite score (selected draft)", f"{data.get('composite_score') or 0:.2f}")
    if data.get("warnings"):
        st.warning("\n".join(data["warnings"]))

    st.subheader("Drafts (one per tone)")
    sit = st.session_state.get("form_situation", situation or "(unspecified)")
    tgt = st.session_state.get("form_target", target)

    draft_images = st.session_state.setdefault("draft_images", {})
    ill_errors = st.session_state.setdefault("illustration_errors", {})

    expand_mode = st.session_state.get(_EXPAND_DRAFT)
    for d in data.get("drafts", []):
        mode = str(d.get("mode", "?"))
        open_expander = expand_mode == mode
        with st.expander(f"{mode}", expanded=open_expander):
            st.write(d.get("text", ""))
            btn_key = f"gen_illu_{mode}"
            if st.button(
                "Generate synthetic illustration (not evidence)",
                key=btn_key,
                help="Requires HUGGINGFACE_API_TOKEN on the server. One image for this draft only.",
            ):
                st.session_state[_PENDING_ILLUSTRATION] = {
                    "mode": mode,
                    "body": {
                        "situation": sit,
                        "target": tgt,
                        "draft_mode": mode,
                        "draft_text": d.get("text", ""),
                    },
                }
                st.rerun()

            if mode in ill_errors:
                st.error(f"Illustration failed: {ill_errors[mode][:500]}")

            img_payload = draft_images.get(mode)
            if img_payload and img_payload.get("image_base64"):
                st.caption(img_payload.get("caption", "") or "")
                raw = base64.b64decode(img_payload["image_base64"])
                st.image(raw, use_container_width=True)

    js = data.get("judge_scores") or {}
    if js:
        st.subheader("Judge scores (all drafts)")
        st.json(js)

    st.caption(f"Request ID: {data.get('request_id', '')}")
