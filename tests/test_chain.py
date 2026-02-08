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
    settings.reranker_enabled = True
    settings.reranker_confidence_threshold = 0.3
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
        assert scores == [1.0, 1.0]
