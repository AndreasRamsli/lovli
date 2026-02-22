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
    index_summary_augmentation_enabled: bool = Field(
        default=False,
        description="Enable index-time summary augmentation (prepends law-level context to provision text).",
    )
    index_summary_catalog_path: str = Field(
        default="data/law_catalog.json",
        description="Path to law catalog JSON for index-time summary augmentation lookup.",
    )
    index_summary_separator: str = Field(
        default="\n\n[LAW_CONTEXT]\n",
        description="Separator inserted between summary prefix and provision text during index augmentation.",
    )
    index_store_raw_augmented_payload: bool = Field(
        default=True,
        description="Store both raw and augmented content variants in Qdrant payload for offline diagnostics.",
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
    trust_profile_name: str = Field(
        default="balanced_v1",
        description="Named trust profile for retrieval/routing/gating defaults (e.g. balanced_v1, strict_v1).",
    )
    trust_profile_version: str = Field(
        default="2026-02-16",
        description="Version tag for the active trust profile.",
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
    reranker_context_enrichment_enabled: bool = Field(
        default=True,
        description="Prepend lightweight metadata context (law name/title/chapter) to reranker document text.",
    )
    reranker_context_max_prefix_chars: int = Field(
        default=300,
        ge=80,
        le=1200,
        description="Maximum prefix length for metadata context prepended to reranker text.",
    )
    reranker_confidence_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Minimum sigmoid-normalized reranker score [0, 1] for confidence gating. "
        "Documents below this threshold trigger a 'no confident answer' response.",
    )
    reranker_min_doc_score: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum per-document reranker score required to keep a document "
        "in the final context.",
    )
    reranker_min_sources: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Minimum number of sources to keep after per-document score filtering.",
    )
    reranker_ambiguity_gating_enabled: bool = Field(
        default=True,
        description="Enable ambiguity gating when top reranker scores are too close.",
    )
    reranker_ambiguity_min_gap: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum gap between top-1 and top-2 reranker scores to consider "
        "the result clearly ranked.",
    )
    reranker_ambiguity_top_score_ceiling: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Only apply ambiguity gating when top score is at or below this ceiling.",
    )
    editorial_notes_per_provision_cap: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum editorial notes attached per provision.",
    )
    editorial_note_max_chars: int = Field(
        default=600,
        ge=100,
        le=5000,
        description="Maximum characters to keep per editorial note payload.",
    )
    editorial_v2_compat_mode: bool = Field(
        default=True,
        description="Allow runtime v2 editorial fetch fallback during migration.",
    )

    # Optional law routing (Tier 0 catalog)
    law_routing_enabled: bool = Field(
        default=False,
        description="Enable lightweight law routing before retrieval using the law catalog.",
    )
    law_routing_reranker_enabled: bool = Field(
        default=False,
        description=(
            "Enable cross-encoder reranker scoring for law-level routing. "
            "When False (default), routing uses embedding similarity + lexical overlap. "
            "bge-reranker-v2-m3 produces near-zero logits for catalog summary pairs, "
            "causing all law scores to collapse to ~0.5 and triggering universal uncertainty "
            "fallback. Disable unless a routing-specific reranker is available."
        ),
    )
    law_routing_embedding_enabled: bool = Field(
        default=True,
        description=(
            "Enable BGE-M3 embedding cosine similarity for law-level routing. "
            "Law routing texts are embedded once at startup and cached. At query time, "
            "the query embedding is compared against all law embeddings to rank candidates. "
            "Blended with lexical token-overlap using law_routing_embedding_weight."
        ),
    )
    law_routing_embedding_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Blend weight for embedding cosine similarity in hybrid law routing. "
            "Final score = embedding_sim * weight + normalized_lexical * (1 - weight). "
            "Higher values favour semantic similarity; lower values favour keyword overlap."
        ),
    )
    law_routing_embedding_text_field: str = Field(
        default="routing_summary_text",
        description=(
            "Which routing text field to embed for law-level similarity scoring. "
            "Options: 'routing_text' (full), 'routing_summary_text' (summary+area+chapters), "
            "'routing_title_text' (title+short_name+ref). "
            "routing_summary_text is recommended: focused, avoids noisy keywords."
        ),
    )
    law_catalog_path: str = Field(
        default="data/law_catalog.json",
        description="Path to the law catalog JSON used for routing.",
    )
    law_routing_max_candidates: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Maximum number of candidate laws to route to before retrieval.",
    )
    law_routing_prefilter_k: int = Field(
        default=80,
        ge=3,
        le=200,
        description=(
            "When embedding routing is enabled: number of top laws returned from the full-catalog "
            "ANN pass before confidence filtering. When embedding is disabled: maximum number of "
            "lexical candidates passed to the cross-encoder reranker."
        ),
    )
    law_routing_rerank_top_k: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Number of top laws to keep after law-level reranker scoring.",
    )
    law_routing_summary_dualpass_enabled: bool = Field(
        default=False,
        description="Score law summary and title/name as separate reranker passages and blend scores.",
    )
    law_routing_dualpass_summary_weight: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Blend weight for summary-specific law reranker score in dual-pass mode.",
    )
    law_routing_dualpass_title_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Blend weight for title/name law reranker score in dual-pass mode.",
    )
    law_routing_dualpass_fulltext_weight: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Blend weight for full routing-text law reranker score in dual-pass mode.",
    )
    law_routing_min_confidence: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Minimum sigmoid-normalized law reranker score required for routed law candidates.",
    )
    law_routing_uncertainty_top_score_ceiling: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="If top routed law score is at or below this ceiling and score gap is small, routing can fall back to broader retrieval.",
    )
    law_routing_uncertainty_min_gap: float = Field(
        default=0.04,
        ge=0.0,
        le=1.0,
        description="Minimum gap between top-1 and top-2 law reranker scores to treat routing as confident.",
    )
    law_routing_fallback_unfiltered: bool = Field(
        default=True,
        description="When routing scores are uncertain, skip law filtering and use unfiltered retrieval.",
    )
    law_routing_fallback_max_laws: int = Field(
        default=12,
        ge=1,
        le=50,
        description="Maximum number of routed laws to keep when uncertainty fallback uses broadened routing instead of unfiltered retrieval.",
    )
    law_routing_fallback_min_lexical_support: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Minimum lexical score support required when building stage-1 broadened fallback law set.",
    )
    law_routing_stage1_min_docs: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Minimum retrieved documents required before accepting stage-1 broadened fallback results.",
    )
    law_routing_stage1_min_top_score: float = Field(
        default=0.32,
        ge=0.0,
        le=1.0,
        description="Minimum top reranker score required before accepting stage-1 broadened fallback results.",
    )
    law_routing_stage1_min_mean_score: float = Field(
        default=0.26,
        ge=0.0,
        le=1.0,
        description="Minimum mean reranker score across top stage-1 docs before accepting stage-1 fallback results.",
    )
    law_routing_min_token_overlap: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Minimum token overlap required for a law to be considered a routing candidate.",
    )
    law_coherence_filter_enabled: bool = Field(
        default=True,
        description="Filter low-confidence singleton sources from non-dominant laws after reranking.",
    )
    law_coherence_min_law_count: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Minimum number of kept sources required for a non-dominant law to remain in context.",
    )
    law_coherence_score_gap: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Minimum average score gap (dominant law - foreign law) required before removing foreign-law singleton sources.",
    )
    law_coherence_relative_gap: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum relative gap between dominant and foreign law strength before filtering non-dominant laws.",
    )
    law_coherence_max_score_weight: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Weight on max score when computing law strength for coherence filtering (remaining weight goes to average score).",
    )
    law_coherence_min_keep: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Minimum number of sources to keep after coherence filtering.",
    )
    law_coherence_dominant_concentration_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="When dominant law occupies at least this share of kept sources, singleton non-dominant laws with no cross-reference affinity are removed aggressively.",
    )
    law_rank_fusion_enabled: bool = Field(
        default=True,
        description="Enable deterministic law-aware rank fusion after reranker + coherence filtering.",
    )
    law_rank_fusion_weight_doc_score: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Rank-fusion weight for document-level cross-encoder score.",
    )
    law_rank_fusion_weight_routing: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Rank-fusion weight for law-level routing alignment.",
    )
    law_rank_fusion_weight_affinity: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Rank-fusion weight for law cross-reference affinity signal.",
    )
    law_rank_fusion_weight_dominance: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Rank-fusion weight for dominant-law coherence context.",
    )
    law_rank_fusion_weight_concentration: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Rank-fusion weight for law concentration among top candidates.",
    )
    law_uncertainty_law_cap_enabled: bool = Field(
        default=True,
        description="Enable temporary law-cap when uncertain fallback yields near-tied multi-law candidates.",
    )
    law_uncertainty_law_cap_max_laws: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Maximum number of laws to keep when uncertainty law-cap is triggered.",
    )
    law_uncertainty_law_cap_max_gap: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum top-2 law strength gap for uncertainty law-cap activation.",
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
            raise ValueError("OPENROUTER_API_KEY is required. Please set it in your .env file.")
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
