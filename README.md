# Alibi AI

**Remote:** [github.com/sinai-abbou/alibi-ai](https://github.com/sinai-abbou/alibi-ai)

**Alibi AI** is a **communication training and simulation** lab. It generates **fictional example** messages (honest, exaggerated, absurd, hypothetical) so you can study tone, narrative, and perceived credibility. Optional outputs include **synthetic, non-verifiable** evidence (mock screenshots, illustrative images) for **UX and learning only**—never real proof.

## Requirements

- Python 3.11+
- [OpenAI API key](https://platform.openai.com/) for text (`gpt-4o-mini` by default)
- Optional: [Hugging Face token](https://huggingface.co/settings/tokens) for image inference; otherwise evidence falls back to local PIL mock images

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: OPENAI_API_KEY=...  (and optionally HUGGINGFACE_API_TOKEN)
```

First API startup downloads the sentence-transformers embedding model (~90MB) and writes `backend/data/embeddings_cache.json` (gitignored).

## Run

**Backend (FastAPI)**

```bash
uvicorn app.api.main:app --reload --app-dir backend --port 8000
```

**Generate endpoint:** `POST /api/generate` with JSON body:

- `situation` (string)
- `tone` (string)
- `target` (string)
- `existing_message` (optional string)
- `generate_evidence` (boolean)

**Streamlit UI**

```bash
export ALIBI_API_BASE=http://127.0.0.1:8000
streamlit run frontend/streamlit_app.py
```

**Docker (API only)**

```bash
docker build -t alibi-ai .
docker run -p 8000:8000 --env-file .env alibi-ai
```

## Development

```bash
ruff check backend frontend
ruff format --check backend frontend
mypy backend/app
pytest
```

## Architecture

- **RAG:** `backend/data/knowledge_cards.json` + `sentence-transformers/all-MiniLM-L6-v2` + cosine similarity; embeddings cached on disk.
- **Agents:** Generator → Risk Analyzer → Judge; deterministic **composite scoring** and **best-draft selection** in `app/agents/judge_scoring.py` (covered by unit tests).
- **Evidence:** planner + PIL mock cards; optional Hugging Face image API with automatic fallback.

## Git workflow

Use **git** and **`gh`** as usual (`pull`, `commit`, `push`). Example:

```bash
git pull origin main
git add -A && git commit -m "feat: your change"
git push origin main
```

`scripts/gh_api_initial_commit.py` is an optional fallback for creating commits via the GitHub REST API when `git` cannot run locally.

## License

Add your license as needed.
