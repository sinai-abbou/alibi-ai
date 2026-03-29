# Alibi AI

**Remote:** [github.com/sinai-abbou/alibi-ai](https://github.com/sinai-abbou/alibi-ai)

Alibi AI is a communication training and simulation project. It generates fictional example messages in five tones:

- `honest`
- `exaggerated`
- `absurd`
- `professional`
- `emotional`

The selected tone determines which draft is highlighted in the UI. The project can also generate optional synthetic illustrations for individual drafts. These images are non-verifiable and intended for learning and interface demonstration only.

## Requirements

- Python 3.11 or later for local development
- Docker for containerized backend runs
- An [OpenAI API key](https://platform.openai.com/) for text generation
- An optional [Hugging Face token](https://huggingface.co/settings/tokens) for on-demand image generation

## Project Structure

- `frontend/streamlit_app.py`: Streamlit user interface
- `backend/app/api/`: FastAPI application and routes
- `backend/app/services/`: orchestration and image-generation services
- `backend/app/rag/`: lightweight RAG retriever
- `backend/data/knowledge_cards.json`: local knowledge base for retrieval

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Then edit `.env` and set at least:

```env
OPENAI_API_KEY=your_key_here
```

Optional settings:

```env
HUGGINGFACE_API_TOKEN=your_token_here
HF_IMAGE_MODEL=stabilityai/stable-diffusion-xl-base-1.0
```

Notes:

- `requirements-dev.txt` is for local development and includes Streamlit and quality tools.
- `requirements.txt` is the backend runtime dependency set used by Docker and Azure.
- On the first backend startup, the embedding model may be downloaded and `backend/data/embeddings_cache.json` may be created locally.

## Local Run

### Backend

```bash
uvicorn app.api.main:app --reload --app-dir backend --port 8000
```

### Frontend

```bash
export ALIBI_API_BASE=http://127.0.0.1:8000
streamlit run frontend/streamlit_app.py
```

## API Overview

### `POST /api/generate`

Request body:

- `situation`: string
- `tone`: one of `honest`, `exaggerated`, `absurd`, `professional`, `emotional`
- `target`: one of `friend`, `family`, `coworker`, `manager`, `client`, `partner`, `teacher_professor`, `formal_official`
- `existing_message`: optional string

### `POST /api/evidence/image`

Generates an optional synthetic illustration for a single draft.

## Docker

The Docker image is backend-only.

```bash
docker build -t alibi-ai .
docker run -p 8000:8000 --env-file .env alibi-ai
```

The image includes only backend runtime dependencies. It does not include Streamlit, pytest, ruff, or mypy.

## Development Commands

```bash
ruff check backend frontend
ruff format --check backend frontend
mypy backend/app
pytest
pytest --cov=backend/app --cov-report=term-missing
```

If you need to recreate your local environment:

```bash
pip install -r requirements-dev.txt
```

The repository also includes an automated quality workflow at `.github/workflows/quality.yml`. It runs Ruff, mypy, pytest, and coverage checks on pushes to `main`, pull requests, and manual runs.

## Azure Deployment

This project deploys the FastAPI backend to Azure App Service using Docker.

Deployment flow:

1. Code is pushed to GitHub.
2. GitHub Actions builds a single-platform `linux/amd64` backend image.
3. The image is pushed to Docker Hub.
4. Azure App Service pulls and runs that image.
5. The local Streamlit frontend connects to the deployed API through `ALIBI_API_BASE`.

The workflow file is:

- `.github/workflows/docker-backend.yml`

Docker Hub image:

- `sinaiabbou/alibi-api:latest`
- `sinaiabbou/alibi-api:sha-<commit>`

Required GitHub repository secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

Recommended Azure app settings:

- `WEBSITES_PORT=8000`
- `OPENAI_API_KEY=...`
- `HUGGINGFACE_API_TOKEN=...` (optional)
- `HF_IMAGE_MODEL=stabilityai/stable-diffusion-xl-base-1.0` (optional)

## Architecture

### Frontend

The frontend is built with Streamlit. It handles:

- user input
- loading states
- draft display
- illustration requests
- interaction with the backend API

### Backend

The backend is built with FastAPI. It handles:

- request validation
- retrieval of relevant knowledge
- message generation
- risk analysis
- judging and scoring
- optional image generation

### RAG

The project uses a lightweight RAG pipeline:

- knowledge cards are stored in `backend/data/knowledge_cards.json`
- they are embedded with `sentence-transformers/all-MiniLM-L6-v2`
- relevant cards are retrieved by cosine similarity
- retrieved chunks are injected into the generation pipeline

Main files:

- `backend/app/rag/retriever.py`
- `backend/app/services/orchestrator.py`

### TDD

The deterministic scoring logic in `backend/app/agents/judge_scoring.py` was developed with a test-first approach and is covered by `backend/tests/test_judge_scoring.py`.

### External Services

- OpenAI: text generation, judging, and scoring
- Hugging Face: optional synthetic image generation

## Git Workflow

Example:

```bash
git pull origin main
git add -A
git commit -m "feat: your change"
git push origin main
```

## License

Add a license if needed.
