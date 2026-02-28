from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central config for the API service.

    - Reads from environment variables and a .env file (if present).
    - Keep defaults sensible for local dev + docker.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Releases / routing
    project_root: Path = Field(default=Path("."), description="Repo root (mounted in docker)")
    releases_dir: Path = Field(default=Path("releases"), description="Directory containing versioned releases")
    active_release_file: Path = Field(default=Path("active_release.txt"))
    default_top_k: int = Field(default=5, ge=1, le=50)

    # A/B split: probability of choosing v2 (keep your current behavior later)
    ab_split_v2: float = Field(default=0.5, ge=0.0, le=1.0)

    # Embedding models per variant (will wire these in later)
    embedding_model_v1: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_model_v2: str = Field(default="sentence-transformers/all-mpnet-base-v2")

    # Telemetry (Langfuse)
    langfuse_enabled: bool = Field(default=True)
    langfuse_public_key: Optional[str] = Field(default=None)
    langfuse_secret_key: Optional[str] = Field(default=None)
    langfuse_host: Optional[str] = Field(default=None)

    # Logging
    log_level: str = Field(default="INFO")

    def release_path(self, variant: str) -> Path:
        return self.releases_dir / variant


def get_settings() -> Settings:
    # Single place to instantiate (easy to override in tests later)
    return Settings()