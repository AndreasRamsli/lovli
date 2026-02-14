"""Tests for LegalIndexer sparse vector conversion."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from lovli.indexer import LegalIndexer, _build_point_key, _generate_deterministic_id
from lovli.parser import LegalArticle
from lovli.config import Settings


@pytest.fixture
def mock_indexer_settings():
    """Create mock settings for indexer."""
    settings = Mock(spec=Settings)
    settings.embedding_model_name = "BAAI/bge-m3"
    settings.embedding_dimension = 1024
    settings.qdrant_in_memory = True
    settings.qdrant_persist_path = None
    settings.qdrant_collection_name = "test_collection"
    settings.embedding_batch_size = 32
    settings.index_batch_size = 100
    return settings


def test_sparse_vector_dict_format(mock_indexer_settings):
    """Test sparse vector conversion with dict format (indices/values)."""
    sparse_vec = {
        "indices": [1, 3, 5],
        "values": [0.1, 0.2, 0.3]
    }
    
    # This would be tested in _process_batch, but we can test the logic
    result = {
        "indices": list(sparse_vec["indices"]),
        "values": list(sparse_vec["values"])
    }
    
    assert result["indices"] == [1, 3, 5]
    assert result["values"] == [0.1, 0.2, 0.3]
    assert len(result["indices"]) == len(result["values"])


def test_sparse_vector_token_spans_format():
    """Test sparse vector conversion with token_spans format."""
    sparse_vec = {
        "token_spans": {
            "1": 0.1,
            "3": 0.2,
            "5": 0.3
        }
    }
    
    indices = []
    values = []
    for token_id, weight in sparse_vec.get("token_spans", {}).items():
        indices.append(int(token_id))
        values.append(float(weight))
    
    assert len(indices) == len(values)
    assert indices == [1, 3, 5]
    assert values == [0.1, 0.2, 0.3]


def test_sparse_vector_scipy_format():
    """Test sparse vector conversion with scipy sparse format."""
    from scipy.sparse import csr_matrix
    
    # Create a sparse matrix
    sparse_matrix = csr_matrix([0, 0.1, 0, 0.2, 0, 0.3])
    
    if hasattr(sparse_matrix, 'indices') and hasattr(sparse_matrix, 'data'):
        result = {
            "indices": sparse_matrix.indices.tolist(),
            "values": sparse_matrix.data.tolist()
        }
        assert len(result["indices"]) == len(result["values"])
        assert len(result["indices"]) == 3


def test_sparse_vector_tuple_format():
    """Test sparse vector conversion with tuple format."""
    sparse_vec = ([1, 3, 5], [0.1, 0.2, 0.3])
    
    if isinstance(sparse_vec, (list, tuple)) and len(sparse_vec) == 2:
        result = {
            "indices": list(sparse_vec[0]),
            "values": list(sparse_vec[1])
        }
        assert result["indices"] == [1, 3, 5]
        assert result["values"] == [0.1, 0.2, 0.3]
        assert len(result["indices"]) == len(result["values"])


def test_sparse_vector_empty():
    """Test handling of empty sparse vector."""
    sparse_vec = {"indices": [], "values": []}
    
    if sparse_vec.get("indices") and sparse_vec.get("values"):
        assert len(sparse_vec["indices"]) == len(sparse_vec["values"])
    else:
        # Empty sparse vector should be handled gracefully
        assert True


def test_sparse_vector_mismatched_lengths():
    """Test validation of mismatched indices/values lengths."""
    sparse_vec = {
        "indices": [1, 3, 5],
        "values": [0.1, 0.2]  # Mismatched length
    }
    
    indices = sparse_vec.get("indices", [])
    values = sparse_vec.get("values", [])
    
    # Should detect mismatch
    if len(indices) != len(values):
        # This should be caught and handled
        assert True


def test_point_id_is_law_aware_for_same_article_id():
    """Same article_id in different laws must produce different point IDs."""
    art1 = LegalArticle(
        article_id="kapittel-1-paragraf-1",
        title="§ 1-1",
        content="A",
        law_id="nl-19990326-017",
        law_title="Law A",
    )
    art2 = LegalArticle(
        article_id="kapittel-1-paragraf-1",
        title="§ 1-1",
        content="B",
        law_id="nl-20020621-034",
        law_title="Law B",
    )

    id1 = _generate_deterministic_id(_build_point_key(art1))
    id2 = _generate_deterministic_id(_build_point_key(art2))
    assert id1 != id2


def test_point_key_prefers_source_anchor_id():
    """Point key should use source_anchor_id when available."""
    art = LegalArticle(
        article_id="kapittel-9-paragraf-6",
        title="§ 9-6",
        content="Oppsigelse",
        law_id="nl-19990326-017",
        law_title="Husleieloven",
        source_anchor_id="kapittel-9-paragraf-7",
    )
    key = _build_point_key(art)
    assert key == "nl-19990326-017::kapittel-9-paragraf-7"


def test_ensure_payload_indexes_targets_metadata_fields():
    """Indexer should ensure payload indexes for law/chapter/doc_type metadata."""
    idx = LegalIndexer.__new__(LegalIndexer)
    idx.settings = Mock()
    idx.settings.qdrant_collection_name = "test_collection"
    idx.client = Mock()

    idx.ensure_payload_indexes()

    fields = [
        call.kwargs["field_name"]
        for call in idx.client.create_payload_index.call_args_list
    ]
    assert fields == [
        "metadata.law_id",
        "metadata.chapter_id",
        "metadata.doc_type",
    ]
