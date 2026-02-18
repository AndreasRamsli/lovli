"""Tests for LegalRAGChain."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from lovli.chain import LegalRAGChain
from lovli.config import Settings


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
    return settings


def test_should_gate_answer_high_score(mock_settings):
    """Test confidence gating with high score."""
    with patch('lovli.chain.QdrantVectorStore'), \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'):
        chain = LegalRAGChain(mock_settings)
        assert chain.should_gate_answer(0.8) is False


def test_should_gate_answer_low_score(mock_settings):
    """Test confidence gating with low score."""
    with patch('lovli.chain.QdrantVectorStore'), \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'):
        chain = LegalRAGChain(mock_settings)
        assert chain.should_gate_answer(0.2) is True


def test_should_gate_answer_no_reranker(mock_settings):
    """Test confidence gating when reranker is disabled."""
    mock_settings.reranker_enabled = False
    with patch('lovli.chain.QdrantVectorStore'), \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'):
        chain = LegalRAGChain(mock_settings)
        assert chain.should_gate_answer(None) is False


def test_validate_question_empty():
    """Test question validation with empty string."""
    with patch('lovli.chain.QdrantVectorStore'), \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'), \
         patch('lovli.chain.get_settings'):
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
    with patch('lovli.chain.QdrantVectorStore'), \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'), \
         patch('lovli.chain.get_settings'):
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
    with patch('lovli.chain.QdrantVectorStore'), \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'):
        chain = LegalRAGChain(mock_settings)
        docs, scores = chain._rerank("test query", [])
        assert docs == []
        assert scores == []


def test_rerank_no_reranker(mock_settings):
    """Test reranking when reranker is disabled."""
    mock_settings.reranker_enabled = False
    with patch('lovli.chain.QdrantVectorStore'), \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'):
        chain = LegalRAGChain(mock_settings)
        mock_docs = [Mock(page_content="doc1"), Mock(page_content="doc2")]
        docs, scores = chain._rerank("test query", mock_docs)
        assert len(docs) == 2
        assert scores == []


def test_apply_reranker_doc_filter_drops_low_scores(mock_settings):
    """Low-scoring docs should be dropped, while preserving min_sources."""
    with patch('lovli.chain.QdrantVectorStore'), \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'):
        chain = LegalRAGChain(mock_settings)
        docs = [Mock(page_content=f"doc{i}") for i in range(4)]
        scores = [0.9, 0.31, 0.2, 0.1]
        filtered_docs, filtered_scores = chain._apply_reranker_doc_filter(docs, scores)
        assert len(filtered_docs) == 2
        assert filtered_scores == [0.9, 0.31]


def test_apply_reranker_doc_filter_keeps_floor(mock_settings):
    """When all scores are low, keep min_sources from the top-ranked list."""
    with patch('lovli.chain.QdrantVectorStore'), \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'):
        chain = LegalRAGChain(mock_settings)
        docs = [Mock(page_content=f"doc{i}") for i in range(3)]
        scores = [0.29, 0.28, 0.27]
        filtered_docs, filtered_scores = chain._apply_reranker_doc_filter(docs, scores)
        assert len(filtered_docs) == mock_settings.reranker_min_sources
        assert filtered_scores == [0.29, 0.28]


def test_should_gate_answer_ambiguity_gap(mock_settings):
    """Gate when top scores are close and below ambiguity ceiling."""
    with patch('lovli.chain.QdrantVectorStore'), \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'):
        chain = LegalRAGChain(mock_settings)
        # Above confidence threshold, but ambiguous due to tiny gap.
        assert chain.should_gate_answer(0.5, scores=[0.5, 0.47, 0.2]) is True


def test_should_not_gate_answer_when_gap_clear(mock_settings):
    """Do not gate when confidence is acceptable and score gap is clear."""
    with patch('lovli.chain.QdrantVectorStore'), \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'):
        chain = LegalRAGChain(mock_settings)
        assert chain.should_gate_answer(0.7, scores=[0.7, 0.5, 0.2]) is False


def test_route_law_ids_matches_catalog_tokens(mock_settings):
    """Routing should prioritize laws with clear lexical overlap."""
    mock_settings.law_routing_enabled = True
    with patch('lovli.chain.QdrantVectorStore'), \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'), \
         patch('lovli.chain.load_catalog') as mock_load_catalog:
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


def test_invoke_retriever_falls_back_when_filtered_empty(mock_settings):
    """If filtered retrieval returns nothing, fallback retriever should run."""
    mock_settings.law_routing_enabled = True
    with patch('lovli.chain.QdrantVectorStore') as mock_vs, \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'), \
         patch('lovli.chain.load_catalog'):
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
    with patch('lovli.chain.QdrantVectorStore'), \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'):
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


def test_prioritize_doc_types_adaptive_budget(mock_settings):
    """Prioritization is a pass-through for attached-editorial model."""
    with patch('lovli.chain.QdrantVectorStore'), \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'):
        chain = LegalRAGChain(mock_settings)
        docs = [
            Mock(metadata={"article_id": "art-1", "doc_type": "provision"}),
            Mock(metadata={"article_id": "art-2", "doc_type": "editorial_note"}),
            Mock(metadata={"article_id": "art-3", "doc_type": "editorial_note"}),
            Mock(metadata={"article_id": "art-4", "doc_type": "editorial_note"}),
        ]
        scores = [0.9, 0.8, 0.7, 0.6]
        selected_docs, selected_scores = chain._prioritize_doc_types(
            docs,
            scores,
            retrieval_query="Hva er reglene for oppsigelse?",
        )
        assert len(selected_docs) == 4
        assert selected_scores == scores


def test_format_context_renders_inline_editorial_notes(mock_settings):
    """Context formatter should render editorial notes inline per provision."""
    with patch('lovli.chain.QdrantVectorStore'), \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'):
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
    with patch('lovli.chain.QdrantVectorStore'), \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'):
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
    with patch('lovli.chain.QdrantVectorStore'), \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'):
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


def test_prioritize_doc_types_no_editorial_edge_case(mock_settings):
    """Prioritization returns empty scores when score/doc lengths mismatch."""
    with patch('lovli.chain.QdrantVectorStore'), \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'):
        chain = LegalRAGChain(mock_settings)
        docs = [
            Mock(metadata={"article_id": "p-1", "doc_type": "provision"}),
            Mock(metadata={"article_id": "p-2", "doc_type": "provision"}),
        ]
        scores = [0.92, 0.9]
        selected_docs, selected_scores = chain._prioritize_doc_types(
            docs,
            [],
            retrieval_query="Hva står i § 9-6?",
        )
        assert [d.metadata["article_id"] for d in selected_docs] == ["p-1", "p-2"]
        assert selected_scores == []


def test_fetch_editorial_for_chapters_fallback_on_filter_error(mock_settings):
    """Editorial fetch should fail open when filtered scroll is unsupported."""
    with patch('lovli.chain.QdrantVectorStore') as mock_vs, \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'):
        vectorstore = MagicMock()
        vectorstore.client = MagicMock()
        vectorstore.client.scroll.side_effect = Exception("Index required but not found")
        mock_vs.return_value = vectorstore
        chain = LegalRAGChain(mock_settings)
        docs = chain._fetch_editorial_for_chapters([("nl-19990326-017", "kapittel-9")])
        assert docs == []


def test_fetch_editorial_for_provisions_fallback_on_filter_error(mock_settings):
    """Linked editorial fetch should fail open when filtered scroll is unsupported."""
    with patch('lovli.chain.QdrantVectorStore') as mock_vs, \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'):
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
    with patch('lovli.chain.QdrantVectorStore') as mock_vs, \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'):
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
                "doc_type": "editorial_note",
            }
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
    with patch('lovli.chain.QdrantVectorStore'), \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'):
        chain = LegalRAGChain(mock_settings)
        p1 = Mock(metadata={"law_id": "nl-19990326-017", "chapter_id": "kapittel-9", "article_id": "p-1", "doc_type": "provision"}, page_content="p1")
        p2 = Mock(metadata={"law_id": "nl-19990326-017", "chapter_id": "kapittel-9", "article_id": "p-2", "doc_type": "provision"}, page_content="p2")
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
    with patch('lovli.chain.QdrantVectorStore'), \
         patch('lovli.chain.HuggingFaceEmbeddings'), \
         patch('lovli.chain.ChatOpenAI'), \
         patch('lovli.chain.QdrantClient'):
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
