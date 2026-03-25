"""Streamlit UI for Alibi AI (wired to FastAPI in later steps)."""

import streamlit as st

st.set_page_config(page_title="Alibi AI", layout="wide")
st.title("Alibi AI")
st.caption("Fictional training examples only — not real-world deception or proof.")

st.info(
    "Configure `ALIBI_API_BASE` (e.g. http://127.0.0.1:8000) and run the FastAPI "
    "backend. Generation UI will be added in a later step."
)
