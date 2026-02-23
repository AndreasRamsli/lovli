"""Tests for LegalRAGChain."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from lovli.chain import LegalRAGChain
from lovli.config import Settings
from lovli.routing import (
    extract_section_article_ids,
    build_law_embedding_index,
    score_all_laws_embedding,
)


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.embedding_model_name = "BAAI/bge-m3"
    settings.embedding_dimension = 1024
    settings.qdrant_in_memory = True
    settings.qdrant_persist_path = None
    settings.qdrant_url = "http://localhost:6333"
    settings.qdrant_api_key = None
    settings.qdrant_collection_name = "test_collection"
    settings.openrouter_api_key = "test_key"
    settings.openrouter_base_url = "https://openrouter.ai/api/v1"
    settings.llm_model = "test-model"
    settings.llm_temperature = 0.1
    settings.retrieval_k = 5
    settings.retrieval_k_initial = 15
    settings.reranker_model = "BAAI/bge-reranker-v2-m3"
    settings.reranker_enabled = False
    settings.reranker_confidence_threshold = 0.3
    settings.reranker_min_doc_score = 0.3
    settings.reranker_min_sources = 2
    settings.reranker_ambiguity_gating_enabled = True
    settings.reranker_ambiguity_min_gap = 0.08
    settings.reranker_ambiguity_top_score_ceiling = 0.65
    settings.editorial_notes_per_provision_cap = 3
    settings.editorial_note_max_chars = 600
    settings.editorial_v2_compat_mode = True
    settings.law_routing_enabled = False
    settings.law_catalog_path = "data/law_catalog.json"
    settings.law_routing_max_candidates = 3
    settings.law_routing_min_token_overlap = 1
    settings.law_routing_prefilter_k = 80
    settings.law_routing_min_confidence = 0.30
    settings.law_routing_rerank_top_k = 6
    settings.law_routing_fallback_max_laws = 12
    settings.law_routing_fallback_min_lexical_support = 1
    settings.law_routing_summary_dualpass_enabled = False
    settings.law_routing_uncertainty_top_score_ceiling = 0.55
    settings.law_routing_uncertainty_min_gap = 0.04
    settings.law_routing_fallback_unfiltered = True
    settings.law_routing_stage1_min_docs = 2
    settings.law_routing_stage1_min_top_score = 0.32
    settings.law_routing_stage1_min_mean_score = 0.26
    settings.law_routing_dualpass_summary_weight = 0.45
    settings.law_routing_dualpass_title_weight = 0.35
    settings.law_routing_dualpass_fulltext_weight = 0.20
    settings.law_routing_reranker_enabled = False
    settings.law_routing_embedding_enabled = False
    settings.law_routing_embedding_text_field = "routing_summary_text"
    return settings


def test_should_gate_answer_high_score(mock_settings):
    """Test confidence gating with high score."""
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        assert chain.should_gate_answer(0.8) is False


def test_should_gate_answer_low_score(mock_settings):
    """Test confidence gating with low score."""
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        assert chain.should_gate_answer(0.2) is True


def test_should_gate_answer_no_reranker(mock_settings):
    """Test confidence gating when reranker is disabled."""
    mock_settings.reranker_enabled = False
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        assert chain.should_gate_answer(None) is False


def test_validate_question_empty():
    """Test question validation with empty string."""
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
        patch("lovli.chain.get_settings"),
    ):
        settings = Mock()
        settings.embedding_model_name = "test"
        settings.qdrant_in_memory = True
        settings.openrouter_api_key = "test"
        settings.llm_model = "test"
        settings.retrieval_k = 5
        settings.retrieval_k_initial = 15
        settings.reranker_enabled = False
        settings.law_routing_enabled = False

        chain = LegalRAGChain(settings)
        with pytest.raises(ValueError, match="Vennligst skriv inn et spørsmål"):
            chain._validate_question("")


def test_validate_question_too_long():
    """Test question validation with overly long question."""
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
        patch("lovli.chain.get_settings"),
    ):
        settings = Mock()
        settings.embedding_model_name = "test"
        settings.qdrant_in_memory = True
        settings.openrouter_api_key = "test"
        settings.llm_model = "test"
        settings.retrieval_k = 5
        settings.retrieval_k_initial = 15
        settings.reranker_enabled = False
        settings.law_routing_enabled = False

        chain = LegalRAGChain(settings)
        long_question = "a" * 2000
        result = chain._validate_question(long_question)
        assert len(result) == 1000  # Should be truncated


def test_rerank_empty_docs(mock_settings):
    """Test reranking with empty document list."""
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        docs, scores = chain._rerank("test query", [])
        assert docs == []
        assert scores == []


def test_rerank_no_reranker(mock_settings):
    """Test reranking when reranker is disabled."""
    mock_settings.reranker_enabled = False
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        mock_docs = [Mock(page_content="doc1"), Mock(page_content="doc2")]
        docs, scores = chain._rerank("test query", mock_docs)
        assert len(docs) == 2
        assert scores == []


def test_apply_reranker_doc_filter_drops_low_scores(mock_settings):
    """Low-scoring docs should be dropped, while preserving min_sources."""
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        docs = [Mock(page_content=f"doc{i}") for i in range(4)]
        scores = [0.9, 0.31, 0.2, 0.1]
        filtered_docs, filtered_scores = chain._apply_reranker_doc_filter(docs, scores)
        assert len(filtered_docs) == 2
        assert filtered_scores == [0.9, 0.31]


def test_apply_reranker_doc_filter_keeps_floor(mock_settings):
    """When all scores are low, keep min_sources from the top-ranked list."""
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        docs = [Mock(page_content=f"doc{i}") for i in range(3)]
        scores = [0.29, 0.28, 0.27]
        filtered_docs, filtered_scores = chain._apply_reranker_doc_filter(docs, scores)
        assert len(filtered_docs) == mock_settings.reranker_min_sources
        assert filtered_scores == [0.29, 0.28]


def test_should_gate_answer_ambiguity_gap(mock_settings):
    """Gate when top scores are close and below ambiguity ceiling."""
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        # Above confidence threshold, but ambiguous due to tiny gap.
        assert chain.should_gate_answer(0.5, scores=[0.5, 0.47, 0.2]) is True


def test_should_not_gate_answer_when_gap_clear(mock_settings):
    """Do not gate when confidence is acceptable and score gap is clear."""
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        assert chain.should_gate_answer(0.7, scores=[0.7, 0.5, 0.2]) is False


def test_route_law_ids_matches_catalog_tokens(mock_settings):
    """Routing should prioritize laws with clear lexical overlap."""
    mock_settings.law_routing_enabled = True
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
        patch("lovli.chain.load_catalog") as mock_load_catalog,
    ):
        mock_load_catalog.return_value = [
            {
                "law_id": "nl-19990326-017",
                "law_title": "Lov om husleieavtaler",
                "law_short_name": "Husleieloven",
                "summary": "Regulerer leie av bolig.",
                "law_ref": "lov/1999-03-26-17",
            },
            {
                "law_id": "nl-20050520-028",
                "law_title": "Lov om arbeidsmiljø",
                "law_short_name": "Arbeidsmiljøloven",
                "summary": "Regulerer arbeidsforhold.",
                "law_ref": "lov/2005-06-17-62",
            },
        ]
        chain = LegalRAGChain(mock_settings)
        routed = chain._route_law_ids("Hva sier husleieloven om depositum?")
        assert routed
        assert routed[0] == "nl-19990326-017"


# ---------------------------------------------------------------------------
# Embedding-based law routing unit tests
# ---------------------------------------------------------------------------


def _make_unit_vec(dim: int, hot: int) -> list[float]:
    """Unit vector with 1.0 at position ``hot`` and 0 elsewhere."""
    v = [0.0] * dim
    v[hot] = 1.0
    return v


def test_score_all_laws_embedding_ranks_by_cosine():
    """Candidates closer to query embedding should rank higher."""
    dim = 4
    catalog = [
        {"law_id": "law-a", "embedding": _make_unit_vec(dim, 0)},
        {"law_id": "law-b", "embedding": _make_unit_vec(dim, 1)},
        {"law_id": "law-c", "embedding": _make_unit_vec(dim, 2)},
    ]
    # Query points toward law-b (dimension 1)
    query_emb = _make_unit_vec(dim, 1)
    result = score_all_laws_embedding(query_emb, catalog, direct_mention_bonus=0.0)
    assert result[0]["law_id"] == "law-b"
    assert result[0]["law_reranker_score"] > result[1]["law_reranker_score"]


def test_score_all_laws_embedding_direct_mention_boost():
    """Direct mention bonus must elevate a law above a closer-embedding competitor."""
    dim = 4
    q = [1.0, 0.0, 0.0, 0.0]
    catalog = [
        # no-bonus: same direction as query but no mention
        {"law_id": "no-bonus", "embedding": q[:], "law_short_name": "Other"},
        # bonus: slightly off-axis but law name appears in query
        {
            "law_id": "with-bonus",
            "embedding": _make_unit_vec(dim, 1),
            "law_short_name": "husleieloven",
            "short_name_normalized": "husleieloven",
        },
    ]
    result = score_all_laws_embedding(
        q,
        catalog,
        direct_mention_bonus=0.20,
        query_norm="husleieloven",
    )
    with_bonus = next(r for r in result if r["law_id"] == "with-bonus")
    assert with_bonus["direct_mention"] is True


def test_score_all_laws_embedding_no_embedding_excluded():
    """Entries without an embedding key must be excluded from results."""
    q = [1.0, 0.0]
    catalog = [
        {"law_id": "no-emb"},
        {"law_id": "has-emb", "embedding": [1.0, 0.0]},
    ]
    result = score_all_laws_embedding(q, catalog, direct_mention_bonus=0.0)
    # Only the entry with an embedding should appear
    assert len(result) == 1
    assert result[0]["law_id"] == "has-emb"


def test_build_law_embedding_index_attaches_embeddings():
    """build_law_embedding_index should attach embedding vectors to each entry."""
    entries = [
        {"law_id": "a", "routing_summary_text": "Husleie regler", "routing_text": "full text a"},
        {"law_id": "b", "routing_summary_text": "Arbeidsrett", "routing_text": "full text b"},
    ]

    def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[float(i)] * 4 for i in range(len(texts))]

    indexed = build_law_embedding_index(entries, fake_embed, text_field="routing_summary_text")
    assert len(indexed) == 2
    assert all("embedding" in e for e in indexed)
    assert len(indexed[0]["embedding"]) == 4


def test_score_all_laws_embedding_empty():
    """Empty catalog list should return empty list without error."""
    result = score_all_laws_embedding([1.0, 0.0], [])
    assert result == []


def test_invoke_retriever_falls_back_when_filtered_empty(mock_settings):
    """If filtered retrieval returns nothing, fallback retriever should run."""
    mock_settings.law_routing_enabled = True
    with (
        patch("lovli.chain.QdrantVectorStore") as mock_vs,
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
        patch("lovli.chain.load_catalog"),
    ):
        mock_vectorstore = MagicMock()
        mock_vs.return_value = mock_vectorstore
        base_retriever = MagicMock()
        base_retriever.invoke.return_value = ["fallback_doc"]
        filtered_retriever = MagicMock()
        filtered_retriever.invoke.return_value = []

        # First call in __init__ creates base retriever; second call in _invoke_retriever creates filtered retriever.
        mock_vectorstore.as_retriever.side_effect = [base_retriever, filtered_retriever]

        chain = LegalRAGChain(mock_settings)
        docs = chain._invoke_retriever("test", routed_law_ids=["nl-19990326-017"])
        assert docs == ["fallback_doc"]


def test_extract_sources_includes_doc_type(mock_settings):
    """Source extraction should carry doc_type for downstream formatting/eval."""
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        doc = Mock(
            metadata={
                "law_id": "nl-19990326-017",
                "law_title": "Husleieloven",
                "article_id": "kapittel-9-paragraf-6",
                "title": "§ 9-6",
                "doc_type": "provision",
            },
            page_content="Oppsigelsesfristen er tre måneder",
        )
        sources = chain._extract_sources([doc], include_content=True)
        assert len(sources) == 1
        assert sources[0]["doc_type"] == "provision"


def test_format_context_renders_inline_editorial_notes(mock_settings):
    """Context formatter should render editorial notes inline per provision."""
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        sources = [
            {
                "law_title": "Husleieloven",
                "article_id": "kapittel-9-paragraf-6",
                "doc_type": "provision",
                "content": "Oppsigelsesfristen er tre måneder.",
                "editorial_notes": [
                    {
                        "article_id": "nl-19990326-017_art_6",
                        "content": "Endret ved lov 16 jan 2009 nr. 6.",
                    }
                ],
            },
        ]
        context = chain._format_context(sources)
        assert "Lovgrunnlag:" in context
        assert "Endringshistorikk:" in context
        assert "[nl-19990326-017_art_6]" in context


def test_attach_editorial_to_provisions_includes_linked_notes(mock_settings):
    """Retrieved provisions should carry attached editorial notes in metadata."""
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        provision_doc = Mock(
            metadata={
                "law_id": "nl-19990326-017",
                "chapter_id": "kapittel-9",
                "article_id": "kapittel-9-paragraf-6",
                "doc_type": "provision",
            },
            page_content="Oppsigelsesfristen er tre måneder.",
        )
        editorial_doc = Mock(
            metadata={
                "law_id": "nl-19990326-017",
                "chapter_id": "kapittel-9",
                "article_id": "nl-19990326-017_kapittel-9_art_6",
                "linked_provision_id": "kapittel-9-paragraf-6",
                "doc_type": "editorial_note",
            },
            page_content="Endret ved lov 16 jan 2009 nr. 6.",
        )
        chain._fetch_editorial_for_provisions = MagicMock(return_value=[editorial_doc])
        attached, attached_scores = chain._attach_editorial_to_provisions([provision_doc])
        assert len(attached) == 1
        assert attached_scores == []
        assert attached[0].metadata["article_id"] == "kapittel-9-paragraf-6"
        notes = attached[0].metadata.get("editorial_notes", [])
        assert len(notes) == 1
        assert "linked_provision_id" not in notes[0]


def test_attach_editorial_to_provisions_keeps_score_alignment(mock_settings):
    """Dropping editorial docs must preserve provision score alignment for gating."""
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        provision_doc = Mock(
            metadata={
                "law_id": "nl-19990326-017",
                "chapter_id": "kapittel-9",
                "article_id": "kapittel-9-paragraf-6",
                "doc_type": "provision",
            },
            page_content="Oppsigelsesfristen er tre måneder.",
        )
        editorial_doc = Mock(
            metadata={
                "law_id": "nl-19990326-017",
                "chapter_id": "kapittel-9",
                "article_id": "nl-19990326-017_kapittel-9_art_6",
                "linked_provision_id": "kapittel-9-paragraf-6",
                "doc_type": "editorial_note",
            },
            page_content="Endret ved lov 16 jan 2009 nr. 6.",
        )
        attached, scores = chain._attach_editorial_to_provisions(
            [provision_doc, editorial_doc],
            scores=[0.91, 0.12],
        )
        assert len(attached) == 1
        assert scores == [0.91]


def test_fetch_editorial_for_chapters_fallback_on_filter_error(mock_settings):
    """Editorial fetch should fail open when filtered scroll is unsupported."""
    with (
        patch("lovli.chain.QdrantVectorStore") as mock_vs,
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        vectorstore = MagicMock()
        vectorstore.client = MagicMock()
        vectorstore.client.scroll.side_effect = Exception("Index required but not found")
        mock_vs.return_value = vectorstore
        chain = LegalRAGChain(mock_settings)
        docs = chain._fetch_editorial_for_chapters([("nl-19990326-017", "kapittel-9")])
        assert docs == []


def test_fetch_editorial_for_provisions_fallback_on_filter_error(mock_settings):
    """Linked editorial fetch should fail open when filtered scroll is unsupported."""
    with (
        patch("lovli.chain.QdrantVectorStore") as mock_vs,
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        vectorstore = MagicMock()
        vectorstore.client = MagicMock()
        vectorstore.client.scroll.side_effect = Exception("Index required but not found")
        mock_vs.return_value = vectorstore
        chain = LegalRAGChain(mock_settings)
        docs = chain._fetch_editorial_for_provisions([("nl-19990326-017", "kapittel-9-paragraf-6")])
        assert docs == []


def test_retrieve_attaches_editorial_notes_before_extracting_sources(mock_settings):
    """Non-reranker path should attach editorial notes to provision metadata."""
    mock_settings.reranker_enabled = False
    with (
        patch("lovli.chain.QdrantVectorStore") as mock_vs,
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        vectorstore = MagicMock()
        retriever = MagicMock()
        provision_doc = Mock(
            metadata={
                "law_id": "nl-19990326-017",
                "chapter_id": "kapittel-9",
                "article_id": "kapittel-9-paragraf-6",
                "doc_type": "provision",
            }
        )
        retriever.invoke.return_value = [provision_doc]
        vectorstore.as_retriever.return_value = retriever
        mock_vs.return_value = vectorstore

        chain = LegalRAGChain(mock_settings)
        editorial_doc = Mock(
            metadata={
                "law_id": "nl-19990326-017",
                "chapter_id": "kapittel-9",
                "article_id": "nl-19990326-017_kapittel-9_art_6",
                "linked_provision_id": "kapittel-9-paragraf-6",
                "doc_type": "editorial_note",
            },
            page_content="Endret ved lov 2010-06-25 nr. 29.",
        )
        chain._fetch_editorial_for_provisions = MagicMock(return_value=[editorial_doc])
        chain._extract_sources = MagicMock(return_value=[])

        chain.retrieve("Hva er oppsigelsestiden?")

        assert chain._fetch_editorial_for_provisions.called
        attached_docs = chain._extract_sources.call_args.args[0]
        assert len(attached_docs) == 1
        notes = attached_docs[0].metadata.get("editorial_notes", [])
        assert len(notes) == 1


def test_attach_editorial_chapter_fallback_skips_ambiguous_chapter(mock_settings):
    """Chapter fallback should not attach notes when multiple provisions share chapter."""
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        p1 = Mock(
            metadata={
                "law_id": "nl-19990326-017",
                "chapter_id": "kapittel-9",
                "article_id": "p-1",
                "doc_type": "provision",
            },
            page_content="p1",
        )
        p2 = Mock(
            metadata={
                "law_id": "nl-19990326-017",
                "chapter_id": "kapittel-9",
                "article_id": "p-2",
                "doc_type": "provision",
            },
            page_content="p2",
        )
        chapter_editorial = Mock(
            metadata={
                "law_id": "nl-19990326-017",
                "chapter_id": "kapittel-9",
                "article_id": "e-1",
                "doc_type": "editorial_note",
            },
            page_content="chapter note",
        )
        chain._fetch_editorial_for_provisions = MagicMock(return_value=[])
        chain._fetch_editorial_for_chapters = MagicMock(return_value=[chapter_editorial])
        attached, _ = chain._attach_editorial_to_provisions([p1, p2], scores=[0.8, 0.7])
        assert attached[0].metadata.get("editorial_notes") == []
        assert attached[1].metadata.get("editorial_notes") == []


def test_attach_editorial_skips_v2_fetch_when_compat_disabled(mock_settings):
    """When compat mode is off, attachment should not trigger runtime fetch methods."""
    mock_settings.editorial_v2_compat_mode = False
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        provision_doc = Mock(
            metadata={
                "law_id": "nl-19990326-017",
                "chapter_id": "kapittel-9",
                "article_id": "kapittel-9-paragraf-6",
                "doc_type": "provision",
            },
            page_content="Oppsigelsesfristen er tre måneder.",
        )
        chain._fetch_editorial_for_provisions = MagicMock(return_value=[])
        chain._fetch_editorial_for_chapters = MagicMock(return_value=[])

        attached, _ = chain._attach_editorial_to_provisions([provision_doc])
        assert len(attached) == 1
        assert attached[0].metadata.get("editorial_notes") == []
        assert not chain._fetch_editorial_for_provisions.called
        assert not chain._fetch_editorial_for_chapters.called


# ---------------------------------------------------------------------------
# _strip_standalone_editorial tests (Fix 4)
# ---------------------------------------------------------------------------


def _make_doc(doc_type: str, article_id: str = "art-1") -> Mock:
    """Helper: create a Mock document with given doc_type in metadata."""
    return Mock(
        metadata={"doc_type": doc_type, "article_id": article_id},
        page_content=f"content for {article_id}",
    )


def test_strip_standalone_editorial_removes_editorial_docs(mock_settings):
    """Editorial note docs must be removed from the ranked list."""
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        provision = _make_doc("provision", "art-1")
        editorial = _make_doc("editorial_note", "art-2")
        docs = [provision, editorial]
        scores = [0.85, 0.72]

        stripped_docs, stripped_scores = chain._strip_standalone_editorial(docs, scores)

        assert len(stripped_docs) == 1
        assert stripped_docs[0] is provision
        assert stripped_scores == [0.85]


def test_strip_standalone_editorial_keeps_all_provisions(mock_settings):
    """When no editorial docs are present, output is unchanged."""
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        docs = [_make_doc("provision", f"art-{i}") for i in range(3)]
        scores = [0.9, 0.8, 0.7]

        stripped_docs, stripped_scores = chain._strip_standalone_editorial(docs, scores)

        assert len(stripped_docs) == 3
        assert stripped_scores == scores


def test_strip_standalone_editorial_preserves_score_alignment(mock_settings):
    """Score list must stay aligned with docs after stripping."""
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        p1 = _make_doc("provision", "art-1")
        ed = _make_doc("editorial_note", "art-e")
        p2 = _make_doc("provision", "art-2")
        docs = [p1, ed, p2]
        scores = [0.88, 0.75, 0.60]

        stripped_docs, stripped_scores = chain._strip_standalone_editorial(docs, scores)

        assert stripped_docs == [p1, p2]
        assert stripped_scores == [0.88, 0.60]


def test_strip_standalone_editorial_degenerate_all_editorial(mock_settings):
    """When every doc is editorial, return the originals rather than an empty list."""
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        docs = [_make_doc("editorial_note", f"art-e{i}") for i in range(2)]
        scores = [0.80, 0.70]

        stripped_docs, stripped_scores = chain._strip_standalone_editorial(docs, scores)

        # Falls back to originals — better than returning empty results
        assert stripped_docs == docs
        assert stripped_scores == scores


# ---------------------------------------------------------------------------
# Ambiguity gate threshold tests for balanced_v2 values (Fix 1 & Fix 2)
# ---------------------------------------------------------------------------


def test_should_gate_answer_ambiguity_gap_balanced_v2(mock_settings):
    """With balanced_v2 min_gap=0.12: gap of 0.08 (< 0.12) should gate."""
    mock_settings.reranker_ambiguity_min_gap = 0.12
    mock_settings.reranker_ambiguity_top_score_ceiling = 0.80
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        # top=0.65, second=0.57 → gap=0.08 < 0.12, top≤0.80 → should gate
        assert chain.should_gate_answer(0.65, scores=[0.65, 0.57, 0.40]) is True


def test_should_not_gate_answer_gap_clear_balanced_v2(mock_settings):
    """With balanced_v2 min_gap=0.12: gap of 0.15 (> 0.12) should not gate."""
    mock_settings.reranker_ambiguity_min_gap = 0.12
    mock_settings.reranker_ambiguity_top_score_ceiling = 0.80
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        # top=0.75, second=0.60 → gap=0.15 > 0.12 → should not gate
        assert chain.should_gate_answer(0.75, scores=[0.75, 0.60, 0.40]) is False


def test_should_not_gate_above_ceiling_balanced_v2(mock_settings):
    """Scores above the 0.80 ceiling are never ambiguity-gated regardless of gap."""
    mock_settings.reranker_ambiguity_min_gap = 0.12
    mock_settings.reranker_ambiguity_top_score_ceiling = 0.80
    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        # top=0.85 > 0.80 ceiling → ambiguity check skipped
        assert chain.should_gate_answer(0.85, scores=[0.85, 0.84, 0.40]) is False


# ---------------------------------------------------------------------------
# balanced_v2 profile presence and values test (Fix 6)
# ---------------------------------------------------------------------------


def test_balanced_v2_profile_values():
    """balanced_v2 must exist and carry the corrected threshold values."""
    from lovli.profiles import TRUST_PROFILES

    assert "balanced_v2" in TRUST_PROFILES, "balanced_v2 profile must be registered"
    p = TRUST_PROFILES["balanced_v2"]

    assert p["reranker_ambiguity_top_score_ceiling"] == 0.80, (
        "top_ceiling must be raised to 0.80 (was 0.70 in balanced_v1)"
    )
    assert p["reranker_ambiguity_min_gap"] == 0.12, (
        "min_gap must be raised to 0.12 (was 0.05 in balanced_v1)"
    )
    assert p["reranker_min_doc_score"] == 0.42, (
        "min_doc_score must be raised to 0.42 (was 0.32 in balanced_v1)"
    )
    assert p["law_routing_summary_dualpass_enabled"] is True, (
        "dualpass must be enabled in balanced_v2"
    )
    # Backward-compatible: balanced_v1 should be unchanged
    v1 = TRUST_PROFILES["balanced_v1"]
    assert v1["reranker_ambiguity_top_score_ceiling"] == 0.70
    assert v1["reranker_ambiguity_min_gap"] == 0.05
    assert v1["reranker_min_doc_score"] == 0.32


# ---------------------------------------------------------------------------
# extract_section_article_ids tests (Fix 1 — article narrowing)
# ---------------------------------------------------------------------------


def test_extract_section_article_ids_dash_pattern():
    """§ X-Y pattern should yield kapittel-X-paragraf-Y."""
    # Use a query that ends with the section ref to avoid trailing letter capture
    result = extract_section_article_ids("Hva sier husleieloven § 3-5?")
    assert result == ["kapittel-3-paragraf-5"]


def test_extract_section_article_ids_letter_suffix():
    """§ X-Ya suffix should be preserved in the output."""
    result = extract_section_article_ids("Husleieloven § 3-5a regulerer")
    assert "kapittel-3-paragraf-5a" in result


def test_extract_section_article_ids_multi():
    """Multiple §§ references in one query should all be extracted."""
    # Use queries where the paragraph number is followed by punctuation
    result = extract_section_article_ids("Se husleieloven §§ 1-1, og § 2-3.")
    assert "kapittel-1-paragraf-1" in result
    assert "kapittel-2-paragraf-3" in result


def test_extract_section_article_ids_no_match():
    """Queries without section references return empty list."""
    result = extract_section_article_ids("Hva er depositumreglene?")
    assert result == []


def test_extract_section_article_ids_empty_query():
    """Empty string returns empty list without error."""
    assert extract_section_article_ids("") == []


def test_extract_section_article_ids_kapittel_pattern():
    """'kapittel X § Y' pattern should yield kapittel-X-paragraf-Y."""
    result = extract_section_article_ids("Se kapittel 2 § 1.")
    assert result == ["kapittel-2-paragraf-1"]


# ---------------------------------------------------------------------------
# build_law_embedding_index with add_title_embedding=True (Fix 2 — dualpass ANN)
# ---------------------------------------------------------------------------


def _make_dummy_embed(dim: int = 4):
    """Return a deterministic embed_documents callable for testing."""
    call_count = [0]

    def embed(texts: list[str]) -> list[list[float]]:
        result = []
        for i, _t in enumerate(texts):
            # Each text gets a distinct unit vector so cosine sims are meaningful
            v = [0.0] * dim
            idx = (call_count[0] + i) % dim
            v[idx] = 1.0
            result.append(v)
        call_count[0] += len(texts)
        return result

    return embed


def test_build_law_embedding_index_adds_title_embedding():
    """add_title_embedding=True must store title_embedding on every entry."""
    entries = [
        {"law_id": "LOV-A", "routing_summary_text": "summary A", "routing_title_text": "title A"},
        {"law_id": "LOV-B", "routing_summary_text": "summary B", "routing_title_text": "title B"},
    ]
    indexed = build_law_embedding_index(
        entries,
        embed_documents=_make_dummy_embed(),
        text_field="routing_summary_text",
        add_title_embedding=True,
    )
    assert len(indexed) == 2
    for entry in indexed:
        assert "embedding" in entry, "summary embedding must be present"
        assert "title_embedding" in entry, (
            "title_embedding must be present when add_title_embedding=True"
        )
        assert len(entry["embedding"]) > 0
        assert len(entry["title_embedding"]) > 0


def test_build_law_embedding_index_no_title_embedding_by_default():
    """Default (add_title_embedding=False) must NOT attach title_embedding."""
    entries = [
        {"law_id": "LOV-A", "routing_summary_text": "summary A", "routing_title_text": "title A"},
    ]
    indexed = build_law_embedding_index(
        entries,
        embed_documents=_make_dummy_embed(),
        text_field="routing_summary_text",
        add_title_embedding=False,
    )
    assert "embedding" in indexed[0]
    assert "title_embedding" not in indexed[0]


# ---------------------------------------------------------------------------
# score_all_laws_embedding with title_weight > 0 (Fix 2 — dualpass blend)
# ---------------------------------------------------------------------------


def test_score_all_laws_embedding_title_blend():
    """title_weight > 0 blends summary and title cosine sims."""
    # Use orthogonal 4-dim unit vectors so sims are deterministic
    query = [1.0, 0.0, 0.0, 0.0]
    # entry 0: summary sim=1.0, title sim=0.0  → blend = 0.7*1 + 0.3*0 = 0.70
    # entry 1: summary sim=0.0, title sim=1.0  → blend = 0.7*0 + 0.3*1 = 0.30
    entries = [
        {
            "law_id": "LOV-A",
            "embedding": [1.0, 0.0, 0.0, 0.0],
            "title_embedding": [0.0, 1.0, 0.0, 0.0],
            "law_title": "Lov A",
            "short_name_normalized": "",
            "law_title_normalized": "lov a",
            "routing_tokens": set(),
        },
        {
            "law_id": "LOV-B",
            "embedding": [0.0, 1.0, 0.0, 0.0],
            "title_embedding": [1.0, 0.0, 0.0, 0.0],
            "law_title": "Lov B",
            "short_name_normalized": "",
            "law_title_normalized": "lov b",
            "routing_tokens": set(),
        },
    ]
    result = score_all_laws_embedding(
        query,
        entries,
        top_k=2,
        direct_mention_bonus=0.0,
        query_norm="",
        title_weight=0.3,
    )
    assert len(result) == 2
    scores = {r["law_id"]: r["law_reranker_score"] for r in result}
    # LOV-A should score higher (0.70) than LOV-B (0.30)
    assert scores["LOV-A"] > scores["LOV-B"]
    assert abs(scores["LOV-A"] - 0.70) < 0.01
    assert abs(scores["LOV-B"] - 0.30) < 0.01


def test_score_all_laws_embedding_no_title_blend_when_missing():
    """When entries lack title_embedding, title_weight is silently ignored."""
    query = [1.0, 0.0, 0.0, 0.0]
    entries = [
        {
            "law_id": "LOV-A",
            "embedding": [1.0, 0.0, 0.0, 0.0],
            # no title_embedding key
            "law_title": "Lov A",
            "short_name": "",
            "routing_tokens": set(),
            "direct_mention_terms": [],
        },
    ]
    # Should not raise, and should fall back to summary-only score
    result = score_all_laws_embedding(
        query,
        entries,
        top_k=1,
        direct_mention_bonus=0.0,
        query_norm="",
        title_weight=0.3,
    )
    assert len(result) == 1
    assert abs(result[0]["law_reranker_score"] - 1.0) < 0.01


# ---------------------------------------------------------------------------
# _build_law_filter with article_id_prefixes (Fix 1 — Qdrant narrowing)
# ---------------------------------------------------------------------------


def test_build_law_filter_with_article_id_prefixes(mock_settings):
    """article_id_prefixes produces nested AND/OR filter clauses."""
    from qdrant_client import models as qdrant_models

    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        law_filter = chain._build_law_filter(
            ["LOV-2000-05-05-25"],
            article_id_prefixes=["kapittel-3-paragraf-5"],
        )
    assert law_filter is not None
    # Should use 'should' with nested Filter clauses (each clause has 'must')
    should_list = getattr(law_filter, "should", None)
    assert should_list, "expected non-empty should list"
    clause = should_list[0]
    # Nested clause is a Filter, which has a 'must' attribute
    assert isinstance(clause, qdrant_models.Filter), "clause must be a nested Filter"
    must_list = getattr(clause, "must", None)
    assert must_list is not None and len(must_list) == 2  # law_id + article_id


def test_build_law_filter_no_prefixes_returns_simple_or(mock_settings):
    """Without article_id_prefixes, produces a flat OR of law_id conditions."""
    from qdrant_client import models as qdrant_models

    with (
        patch("lovli.chain.QdrantVectorStore"),
        patch("lovli.chain.HuggingFaceEmbeddings"),
        patch("lovli.chain.ChatOpenAI"),
        patch("lovli.chain.QdrantClient"),
    ):
        chain = LegalRAGChain(mock_settings)
        law_filter = chain._build_law_filter(["LOV-A", "LOV-B"], article_id_prefixes=None)
    assert law_filter is not None
    should_list = getattr(law_filter, "should", None)
    assert should_list is not None and len(should_list) == 2
    # Each clause is a FieldCondition (not a nested Filter)
    assert isinstance(should_list[0], qdrant_models.FieldCondition)
