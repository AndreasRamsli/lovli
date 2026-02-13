"""Integration tests for hybrid search with Qdrant."""

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
)
from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings


class DummyEmbeddings(Embeddings):
    """Minimal valid LangChain embeddings for testing."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4]] * len(texts)
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


@pytest.fixture
def qdrant_client():
    """Create an in-memory Qdrant client for testing."""
    return QdrantClient(":memory:")


def _insert_named_vectors(client: QdrantClient, collection: str):
    """Insert sample points with named dense + sparse vectors."""
    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(
                id=1,
                vector={
                    "dense": [0.1, 0.2, 0.3, 0.4],
                    "sparse": SparseVector(indices=[0, 1, 2], values=[0.5, 0.3, 0.1]),
                },
                payload={
                    "page_content": "Depositum skal ikke overstige seks maaneders leie",
                    "metadata": {
                        "article_id": "kapittel-3-paragraf-5",
                        "title": "Depositum",
                        "law_id": "nl-test",
                        "law_title": "Husleieloven",
                        "doc_type": "provision",
                        "url": "https://lovdata.no/lov/test#k3-p5",
                    },
                },
            ),
            PointStruct(
                id=2,
                vector={
                    "dense": [0.2, 0.3, 0.4, 0.5],
                    "sparse": SparseVector(indices=[1, 2, 3], values=[0.6, 0.2, 0.1]),
                },
                payload={
                    "page_content": "Oppsigelsestiden er tre maaneder",
                    "metadata": {
                        "article_id": "kapittel-9-paragraf-6",
                        "title": "Oppsigelse",
                        "law_id": "nl-test",
                        "law_title": "Husleieloven",
                        "doc_type": "provision",
                        "url": "https://lovdata.no/lov/test#k9-p6",
                    },
                },
            ),
            PointStruct(
                id=3,
                vector={
                    "dense": [0.3, 0.4, 0.5, 0.6],
                    "sparse": SparseVector(indices=[2, 3, 4], values=[0.4, 0.3, 0.2]),
                },
                payload={
                    "page_content": "Leier skal vedlikeholde doerlaaser og kraner",
                    "metadata": {
                        "article_id": "kapittel-5-paragraf-3",
                        "title": "Vedlikehold",
                        "law_id": "nl-test",
                        "law_title": "Husleieloven",
                        "doc_type": "provision",
                        "url": "https://lovdata.no/lov/test#k5-p3",
                    },
                },
            ),
        ],
    )


class TestNamedVectorsWithQdrant:
    """Test that named vectors format works with Qdrant client."""

    def test_create_collection_named_dense_and_sparse(self, qdrant_client):
        """Collection must use named dense vectors when adding sparse vectors."""
        qdrant_client.create_collection(
            collection_name="test_hybrid",
            vectors_config={"dense": VectorParams(size=4, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
        info = qdrant_client.get_collection("test_hybrid")
        assert info is not None

    def test_upsert_named_vectors(self, qdrant_client):
        """Named vectors can be upserted."""
        qdrant_client.create_collection(
            collection_name="test_hybrid",
            vectors_config={"dense": VectorParams(size=4, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
        _insert_named_vectors(qdrant_client, "test_hybrid")
        info = qdrant_client.get_collection("test_hybrid")
        assert info.points_count == 3

    def test_dense_query(self, qdrant_client):
        """Dense-only query works against named vectors collection."""
        qdrant_client.create_collection(
            collection_name="test_hybrid",
            vectors_config={"dense": VectorParams(size=4, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
        _insert_named_vectors(qdrant_client, "test_hybrid")

        results = qdrant_client.query_points(
            collection_name="test_hybrid",
            query=[0.1, 0.2, 0.3, 0.4],
            using="dense",
            limit=2,
        )
        assert len(results.points) == 2


class TestQdrantVectorStoreWithNamedVectors:
    """Test QdrantVectorStore works with named vectors via vector_name parameter."""

    def test_similarity_search(self, qdrant_client):
        """QdrantVectorStore.similarity_search works with vector_name='dense'."""
        qdrant_client.create_collection(
            collection_name="test_hybrid",
            vectors_config={"dense": VectorParams(size=4, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
        _insert_named_vectors(qdrant_client, "test_hybrid")

        vs = QdrantVectorStore(
            client=qdrant_client,
            collection_name="test_hybrid",
            embedding=DummyEmbeddings(),
            vector_name="dense",
        )
        results = vs.similarity_search("test", k=2)
        assert len(results) == 2
        assert results[0].page_content  # Not empty
        assert "article_id" in results[0].metadata

    def test_as_retriever(self, qdrant_client):
        """QdrantVectorStore.as_retriever works with named vectors."""
        qdrant_client.create_collection(
            collection_name="test_hybrid",
            vectors_config={"dense": VectorParams(size=4, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
        _insert_named_vectors(qdrant_client, "test_hybrid")

        vs = QdrantVectorStore(
            client=qdrant_client,
            collection_name="test_hybrid",
            embedding=DummyEmbeddings(),
            vector_name="dense",
        )
        retriever = vs.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke("test query")
        assert len(docs) == 2
        assert docs[0].page_content
        assert "article_id" in docs[0].metadata

    def test_dense_only_collection_no_vector_name(self, qdrant_client):
        """Dense-only collections work without vector_name (backward compat)."""
        qdrant_client.create_collection(
            collection_name="test_dense",
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )
        qdrant_client.upsert(
            collection_name="test_dense",
            points=[
                PointStruct(
                    id=1,
                    vector=[0.1, 0.2, 0.3, 0.4],
                    payload={"page_content": "test content", "metadata": {"id": "1"}},
                ),
            ],
        )

        vs = QdrantVectorStore(
            client=qdrant_client,
            collection_name="test_dense",
            embedding=DummyEmbeddings(),
        )
        results = vs.similarity_search("test", k=1)
        assert len(results) == 1
