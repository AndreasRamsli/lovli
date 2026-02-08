"""Configuration management using Pydantic settings."""

from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenRouter API Configuration
    openrouter_api_key: str = Field(
        ...,
        description="OpenRouter API key for LLM access",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
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
        description="OpenRouter model name (e.g., z-ai/glm-4.7-flash, z-ai/glm-4.7, openai/gpt-4o-mini)",
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
        description="Number of documents to retrieve for RAG (final count after reranking)",
    )
    retrieval_k_initial: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Number of documents to retrieve before reranking (over-retrieve for reranker)",
    )
    
    # Reranker Configuration
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Cross-encoder reranker model for rescoring retrieved documents",
    )
    reranker_enabled: bool = Field(
        default=True,
        description="Enable cross-encoder reranking",
    )
    reranker_confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum reranker score threshold for confidence gating",
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

    # LangSmith Evaluation Configuration
    langsmith_project: str | None = Field(
        default="lovli-evals",
        description="LangSmith project name for evaluation experiments",
    )
    eval_judge_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model to use as LLM-as-judge for evaluations (e.g., gpt-4o-mini, gpt-4.1-mini)",
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
