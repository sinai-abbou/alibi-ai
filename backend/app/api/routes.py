"""HTTP routes."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from openai import APIError

from app.api.deps import get_openai, get_retriever
from app.rag.retriever import KnowledgeRetriever
from app.schemas import GenerateRequest, GenerateResponse
from app.services.orchestrator import run_pipeline
from app.utils.logging import get_logger, set_correlation_id
from app.utils.openai_client import OpenAIClient
from app.utils.settings import Settings, get_settings

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["api"])


@router.post("/generate", response_model=GenerateResponse)
def generate_messages(
    body: GenerateRequest,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    retriever: Annotated[KnowledgeRetriever, Depends(get_retriever)],
    client: Annotated[OpenAIClient, Depends(get_openai)],
    x_request_id: Annotated[str | None, Header(alias="X-Request-ID")] = None,
) -> GenerateResponse:
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY missing")
        raise HTTPException(status_code=503, detail="OpenAI API key not configured.")
    rid = x_request_id or getattr(request.state, "request_id", None) or str(uuid.uuid4())
    set_correlation_id(rid)
    try:
        return run_pipeline(
            request=body,
            settings=settings,
            retriever=retriever,
            client=client,
            request_id=rid,
        )
    except APIError as e:
        logger.exception("OpenAI API error for request %s", rid)
        raise HTTPException(status_code=502, detail=f"Upstream LLM error: {e}") from e
    finally:
        set_correlation_id(None)
