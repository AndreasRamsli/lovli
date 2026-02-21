"""Lovli - Legal RAG assistant for Norwegian laws."""

__version__ = "0.1.0"

# Core chain
from .chain import LegalRAGChain, GATED_RESPONSE, NO_RESULTS_RESPONSE

# Types
from .types import RoutingResult, RetrievalResult

# Scoring / retrieval utilities
from .scoring import (
    sigmoid,
    normalize_sigmoid_scores,
    matches_expected_source,
    normalize_law_ref,
    build_law_cross_reference_affinity,
    build_law_coherence_decision,
    build_law_aware_rank_fusion,
    apply_uncertainty_law_cap,
    _infer_doc_type,
)

# Routing utilities
from .routing import (
    build_routing_entries,
    score_law_candidates_lexical,
    score_law_candidates_reranker,
    compute_routing_alignment,
    tokenize_for_routing,
    normalize_keyword_term,
    normalize_law_mention,
)

# Reranking utilities
from .reranking import (
    build_reranker_document_text,
    rerank_documents,
    filter_reranked_docs,
    should_gate_answer,
)

# Profiles / trust
from .profiles import (
    TRUST_PROFILES,
    apply_trust_profile,
    extract_chat_history,
)

# Evaluation utilities
from .eval import (
    collect_indexed_article_ids,
    validate_questions,
    infer_negative_type,
)

__all__ = [
    # chain
    "LegalRAGChain",
    "GATED_RESPONSE",
    "NO_RESULTS_RESPONSE",
    # types
    "RoutingResult",
    "RetrievalResult",
    # scoring
    "sigmoid",
    "normalize_sigmoid_scores",
    "matches_expected_source",
    "normalize_law_ref",
    "build_law_cross_reference_affinity",
    "build_law_coherence_decision",
    "build_law_aware_rank_fusion",
    "apply_uncertainty_law_cap",
    "_infer_doc_type",
    # routing
    "build_routing_entries",
    "score_law_candidates_lexical",
    "score_law_candidates_reranker",
    "compute_routing_alignment",
    "tokenize_for_routing",
    "normalize_keyword_term",
    "normalize_law_mention",
    # reranking
    "build_reranker_document_text",
    "rerank_documents",
    "filter_reranked_docs",
    "should_gate_answer",
    # profiles
    "TRUST_PROFILES",
    "apply_trust_profile",
    "extract_chat_history",
    # eval
    "collect_indexed_article_ids",
    "validate_questions",
    "infer_negative_type",
]
