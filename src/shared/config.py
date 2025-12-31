"""Configuration management for RAG Brain."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    postgres_url: str = "postgresql+asyncpg://postgres:ragbrain@localhost:5433/workbench"

    # Ollama
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768

    # Agent Mail
    agent_mail_url: str = "http://localhost:8765"

    # Training
    training_threshold: int = 500
    min_quality_threshold: float = 0.3

    # Ranking weights
    similarity_weight: float = 0.4
    quality_weight: float = 0.25
    usefulness_weight: float = 0.25
    recency_weight: float = 0.1

    # Tier thresholds
    core_quality_threshold: float = 0.8
    core_usefulness_threshold: float = 0.6
    core_access_threshold: int = 3
    quarantine_quality_threshold: float = 0.3
    quarantine_usefulness_threshold: float = 0.3
    archive_days_inactive: int = 90

    # Duplicate detection
    duplicate_similarity_threshold: float = 0.95

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
