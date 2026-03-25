# Alibi AI

Multi-agent pipeline for **training and simulation**: it produces **fictional example** message drafts and optional **synthetic, non-verifiable** evidence for UX and learning. Nothing here is intended as real-world deception or proof.

## Requirements

- Python 3.11+
- [OpenAI API key](https://platform.openai.com/) for text generation (`gpt-4o-mini` in later steps)
- Optional: Hugging Face token for image APIs (evidence module can fall back to local PIL mocks)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (and optionally HUGGINGFACE_API_TOKEN)
```

## Run (after implementation steps)

- **API:** from repo root, `uvicorn app.api.main:app --reload --app-dir backend`
- **Streamlit:** `streamlit run frontend/streamlit_app.py`
- **Docker:** `docker build -t alibi-ai .` then `docker run -p 8000:8000 --env-file .env alibi-ai`

Configure the Streamlit client with `ALIBI_API_BASE` (default `http://127.0.0.1:8000`).

## Development

```bash
ruff check backend frontend
ruff format --check backend frontend
mypy backend/app
pytest
```

## License

Add your license as needed.
