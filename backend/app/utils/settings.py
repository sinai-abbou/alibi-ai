"""Application settings from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = Field(default="", description="OpenAI API key for text generation.")
    openai_model: str = "gpt-4o-mini"

    huggingface_api_token: str | None = Field(default=None)

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_top_k: int = 5

    knowledge_cards_path: Path = Field(
        default_factory=lambda: (
            Path(__file__).resolve().parent.parent.parent / "data" / "knowledge_cards.json"
        ),
    )
    embeddings_cache_path: Path = Field(
        default_factory=lambda: (
            Path(__file__).resolve().parent.parent.parent / "data" / "embeddings_cache.json"
        ),
    )

    hf_image_model: str = "stabilityai/stable-diffusion-xl-base-1.0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
