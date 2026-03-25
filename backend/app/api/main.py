"""FastAPI application entry."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from app.api.routes import router
from app.rag.retriever import KnowledgeRetriever
from app.utils.logging import configure_logging, get_logger
from app.utils.settings import get_settings

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()
    retriever = KnowledgeRetriever(settings)
    logger.info("Warming RAG retriever (may download embedding model on first run)...")
    retriever.warm()
    app.state.retriever = retriever
    yield


app = FastAPI(title="Alibi AI", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
