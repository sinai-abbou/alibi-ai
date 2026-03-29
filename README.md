# Alibi AI

**Remote:** [github.com/sinai-abbou/alibi-ai](https://github.com/sinai-abbou/alibi-ai)

**Alibi AI** is a **communication training and simulation** lab. It generates **fictional example** messages in five tones (honest, exaggerated, absurd, professional, emotional); the UI tone selects which draft is highlighted. **Synthetic illustrations** (optional, on-demand per draft via the API or Streamlit buttons) are **non-verifiable** and for **UX and learning only**—never real proof.

## Requirements

- Python 3.11+
- [OpenAI API key](https://platform.openai.com/) for text (`gpt-4o-mini` by default)
- Optional: [Hugging Face token](https://huggingface.co/settings/tokens) for **on-demand** image generation (`POST /api/evidence/image`). Free-tier Inference Providers have rate limits; quality varies by model and queue time.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env
# Edit .env: OPENAI_API_KEY=...  (and optionally HUGGINGFACE_API_TOKEN)
```

First API startup downloads the sentence-transformers embedding model (~90MB) and writes `backend/data/embeddings_cache.json` (gitignored).

`requirements.txt` is now reserved for the **backend runtime image** (Docker / Azure). Local development uses `requirements-dev.txt`, which includes Streamlit plus test/lint/type-check tooling.

## Run

**Backend (FastAPI)**

```bash
uvicorn app.api.main:app --reload --app-dir backend --port 8000
```

**Generate endpoint:** `POST /api/generate` with JSON body:

- `situation` (string)
- `tone` — one of: `honest`, `exaggerated`, `absurd`, `professional`, `emotional`
- `target` — one of: `friend`, `family`, `coworker`, `manager`, `client`, `partner`, `teacher_professor`, `formal_official`
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

This Docker image installs **backend runtime dependencies only**. It does **not** include Streamlit, pytest, ruff, or mypy.

## Development

```bash
ruff check backend frontend
ruff format --check backend frontend
mypy backend/app
pytest
```

If you need to reinstall your local environment, use:

```bash
pip install -r requirements-dev.txt
```

## Azure / Docker Hub deployment

The repo includes a GitHub Actions workflow at `.github/workflows/docker-backend.yml` that builds an **Azure App Service-compatible** backend image and pushes it to Docker Hub:

- image: `sinaiabbou/alibi-api:latest`
- platform: `linux/amd64` only
- provenance / SBOM disabled to avoid multi-arch / OCI-index style output

Add these GitHub repository secrets before using the workflow:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

Typical flow:

1. Push to `main` or run the workflow manually.
2. GitHub Actions builds and pushes `sinaiabbou/alibi-api:latest`.
3. Azure App Service pulls that image.
4. Local Streamlit points to Azure with `ALIBI_API_BASE=https://<your-app>.azurewebsites.net`.

Recommended Azure app settings:

- `WEBSITES_PORT=8000`
- `OPENAI_API_KEY=...`
- `HUGGINGFACE_API_TOKEN=...` (optional)
- `HF_IMAGE_MODEL=stabilityai/stable-diffusion-xl-base-1.0` (optional)

## Architecture

- **RAG:** `backend/data/knowledge_cards.json` + `sentence-transformers/all-MiniLM-L6-v2` + cosine similarity; embeddings cached on disk.
- **Agents:** Generator → Risk Analyzer → Judge; deterministic **composite scoring** and **best-draft selection** in `app/agents/judge_scoring.py` (covered by unit tests).
- **Evidence:** Hugging Face image generation only via `POST /api/evidence/image` (not bundled with `/api/generate`). Streamlit offers a button under each draft expander.

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
