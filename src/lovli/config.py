"""Configuration management using Pydantic settings."""

from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenRouter API Configuration
    openrouter_api_key: str = Field("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",
        description="OpenRouter API key for LLM access",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )

    # Hugging Face API (optional, for embeddings)
    hf_api_token: str | None = Field(
        default=None,
        description="Hugging Face API token (optional, for API-based embeddings)",
    )

    # Qdrant Configuration
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL (ignored if qdrant_in_memory is True)",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Qdrant API key (required for cloud instances)",
    )
    qdrant_collection_name: str = Field(
        default="lovli_laws",
        description="Qdrant collection name for storing legal articles",
    )
    qdrant_in_memory: bool = Field(
        default=False,
        description="Use in-memory Qdrant (for testing, no Docker needed)",
    )
    qdrant_persist_path: str | None = Field(
        default=None,
        description="Path to persist Qdrant data (only for in-memory mode)",
    )

    # Embedding Model Configuration
    embedding_model_name: str = Field(
        default="BAAI/bge-m3",
        description="Hugging Face model name for embeddings",
    )
    embedding_dimension: int = Field(
        default=1024,
        description="Embedding vector dimension (BGE-M3 uses 1024)",
    )

    # LLM Configuration (via OpenRouter)
    llm_model: str = Field(
        default="z-ai/glm-4.7-flash",
        description="OpenRouter model name (e.g., z-ai/glm-4.7, openai/gpt-4o-mini)",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature (lower = more factual)",
    )

    # Retrieval Configuration
    retrieval_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve for RAG",
    )

    # Data Paths (relative to project root)
    data_nl_path: str = Field(
        default="data/nl",
        description="Path to Norwegian laws directory",
    )
    data_sf_path: str = Field(
        default="data/sf",
        description="Path to regulations directory",
    )

    # Indexing Configuration
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for embedding generation (memory dependent)",
    )
    index_batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Batch size for indexing articles",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("openrouter_api_key")
    @classmethod
    def validate_openrouter_key(cls, v: str) -> str:
        """Validate that OpenRouter API key is provided."""
        if not v or v.strip() == "":
            raise ValueError(
                "OPENROUTER_API_KEY is required. Please set it in your .env file."
            )
        return v.strip()

    def get_data_nl_path(self) -> Path:
        """Get absolute path to Norwegian laws directory."""
        return Path(self.data_nl_path).resolve()

    def get_data_sf_path(self) -> Path:
        """Get absolute path to regulations directory."""
        return Path(self.data_sf_path).resolve()


# Singleton instance
_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """
    Get application settings instance (singleton pattern).

    Returns:
        Settings instance (cached after first call)
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
