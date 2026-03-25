"""FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from app.rag.retriever import KnowledgeRetriever
from app.utils.openai_client import OpenAIClient
from app.utils.settings import Settings, get_settings


def get_retriever(request: Request) -> KnowledgeRetriever:
    return cast(KnowledgeRetriever, request.app.state.retriever)


def get_openai(settings: Annotated[Settings, Depends(get_settings)]) -> OpenAIClient:
    return OpenAIClient(settings)
