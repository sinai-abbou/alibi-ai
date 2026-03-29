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

    hf_image_model: str = Field(
        default="stabilityai/stable-diffusion-xl-base-1.0",
        description="HF Hub model id for text-to-image. Env: HF_IMAGE_MODEL.",
    )
    hf_image_guidance_scale: float = Field(
        default=7.5,
        ge=1.0,
        le=20.0,
        description="Text-to-image guidance. Env: HF_IMAGE_GUIDANCE_SCALE.",
    )
    hf_image_num_inference_steps: int = Field(
        default=32,
        ge=1,
        le=80,
        description=(
            "Diffusion steps (if supported by provider). "
            "Env: HF_IMAGE_NUM_INFERENCE_STEPS."
        ),
    )
    hf_image_width: int = Field(
        default=1024,
        ge=512,
        le=1536,
        description="Output width in pixels (if supported). Env: HF_IMAGE_WIDTH.",
    )
    hf_image_height: int = Field(
        default=1024,
        ge=512,
        le=1536,
        description="Output height in pixels (if supported). Env: HF_IMAGE_HEIGHT.",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
