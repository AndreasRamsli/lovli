"""Vector indexing module for storing legal articles in Qdrant."""

import hashlib
import logging
from typing import Iterator
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from .config import Settings
from .parser import LegalArticle

logger = logging.getLogger(__name__)


def _build_point_key(article: LegalArticle) -> str:
    """
    Build stable, law-aware key for deterministic Qdrant point IDs.

    Prefer source_anchor_id when available so key reflects source-level identity.
    """
    stable_source_id = article.source_anchor_id or article.article_id
    return f"{article.law_id}::{stable_source_id}"


def _generate_deterministic_id(point_key: str) -> int:
    """
    Generate a deterministic int64 ID from article ID string.

    Uses SHA-256 hash to avoid collisions while ensuring determinism.

    Args:
        point_key: Unique, law-aware point key

    Returns:
        int64 ID suitable for Qdrant
    """
    # Use SHA-256 for deterministic hashing
    hash_bytes = hashlib.sha256(point_key.encode("utf-8")).digest()
    # Take first 8 bytes and convert to int64 (unsigned to ensure non-negative)
    return int.from_bytes(hash_bytes[:8], byteorder="big", signed=False) % (2**63)


class LegalIndexer:
    """Handles indexing of legal articles into Qdrant vector database."""

    def __init__(self, settings: Settings | None = None):
        """
        Initialize the indexer with settings and embedding model.

        Args:
            settings: Application settings (defaults to loading from env)

        Raises:
            ValueError: If embedding model cannot be loaded
            ConnectionError: If Qdrant connection fails
        """
        from .config import get_settings

        self.settings = settings or get_settings()

        try:
            if self.settings.qdrant_in_memory:
                # Use in-memory Qdrant (for testing without Docker)
                if self.settings.qdrant_persist_path:
                    self.client = QdrantClient(path=self.settings.qdrant_persist_path)
                    logger.info(f"Using Qdrant with persistence at {self.settings.qdrant_persist_path}")
                else:
                    self.client = QdrantClient(":memory:")
                    logger.info("Using in-memory Qdrant (data will not persist)")
            else:
                self.client = QdrantClient(
                    url=self.settings.qdrant_url,
                    api_key=self.settings.qdrant_api_key,
                )
                logger.info(f"Connected to Qdrant at {self.settings.qdrant_url}")
            # Test connection
            self.client.get_collections()
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise ConnectionError(f"Cannot connect to Qdrant: {e}") from e

        # Detect BGE-M3 model
        self._is_bge_m3 = "bge-m3" in self.settings.embedding_model_name.lower()
        self._use_native_api = False
        self._flag_model = None

        # Try to load BGEM3FlagModel for native dense+sparse support
        if self._is_bge_m3:
            try:
                from FlagEmbedding import BGEM3FlagModel

                logger.info(f"Loading BGE-M3 via FlagEmbedding (native API): {self.settings.embedding_model_name}")
                self._flag_model = BGEM3FlagModel(
                    self.settings.embedding_model_name,
                    use_fp16=True,
                )
                self._use_native_api = True
                logger.info("BGE-M3 loaded with native API (dense + sparse support)")
            except ImportError:
                logger.info("FlagEmbedding not installed, falling back to SentenceTransformer (dense-only)")
            except Exception as e:
                logger.warning(f"Failed to load BGEM3FlagModel: {e}. Falling back to SentenceTransformer.")

        # Fall back to SentenceTransformer for dense-only encoding
        if not self._use_native_api:
            try:
                logger.info(f"Loading embedding model: {self.settings.embedding_model_name}")
                # Force CPU to avoid MPS (Apple Silicon GPU) out-of-memory errors
                # on large batches. CPU is reliable and fast enough for indexing.
                self.embedding_model = SentenceTransformer(
                    self.settings.embedding_model_name,
                    device="cpu",
                )
                logger.info("Embedding model loaded successfully (dense-only, CPU)")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise ValueError(f"Cannot load embedding model: {e}") from e
        else:
            self.embedding_model = None  # Not needed when using native API

        # Track actual sparse support based on what we loaded
        self._has_sparse = self._use_native_api

    def collection_exists(self, collection_name: str | None = None) -> bool:
        """
        Check if a collection exists in Qdrant.

        Args:
            collection_name: Name of the collection (defaults to settings value)

        Returns:
            True if collection exists, False otherwise
        """
        name = collection_name or self.settings.qdrant_collection_name
        try:
            return self.client.collection_exists(collection_name=name)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False

    def create_collection(
        self, collection_name: str | None = None, recreate: bool = False
    ) -> None:
        """
        Create a Qdrant collection for storing legal articles.

        Args:
            collection_name: Name of the collection (defaults to settings value)
            recreate: If True, delete existing collection before creating

        Raises:
            ValueError: If collection already exists and recreate=False
        """
        name = collection_name or self.settings.qdrant_collection_name

        if self.collection_exists(name):
            if recreate:
                logger.warning(f"Deleting existing collection: {name}")
                self.client.delete_collection(collection_name=name)
            else:
                logger.info(f"Collection {name} already exists, skipping creation")
                return

        logger.info(f"Creating collection: {name}")

        if self._has_sparse:
            # Use named dense vector + named sparse vector for hybrid search.
            # Named vectors are required when combining dense and sparse in Qdrant.
            vectors_config = {"dense": VectorParams(
                size=self.settings.embedding_dimension,
                distance=Distance.COSINE,
            )}
            sparse_vectors_config = {"sparse": SparseVectorParams()}
            logger.info("Configuring named vectors (dense + sparse) for hybrid search")
        else:
            # Dense-only: use default (unnamed) vector
            vectors_config = VectorParams(
                size=self.settings.embedding_dimension,
                distance=Distance.COSINE,
            )
            sparse_vectors_config = None
        
        self.client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
            on_disk_payload=True,  # Store payloads on disk to save RAM
        )
        logger.info(f"Collection {name} created successfully (payloads on disk)")

    def ensure_payload_indexes(self, collection_name: str | None = None) -> None:
        """
        Ensure payload keyword indexes needed for filtering are present.

        This operation is safe to run repeatedly.
        """
        name = collection_name or self.settings.qdrant_collection_name
        fields = [
            "metadata.law_id",
            "metadata.chapter_id",
            "metadata.doc_type",
        ]
        for field_name in fields:
            try:
                self.client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema="keyword",
                    wait=True,
                )
                logger.info("Ensured payload index: %s", field_name)
            except Exception as exc:
                # Keep this idempotent; if index already exists or API differs, continue.
                logger.warning("Failed ensuring payload index %s: %s", field_name, exc)

    def index_articles(self, articles: Iterator[LegalArticle]) -> int:
        """
        Index a batch of legal articles into Qdrant.

        Args:
            articles: Iterator of LegalArticle objects to index

        Returns:
            Number of articles indexed

        Raises:
            ValueError: If collection doesn't exist
            RuntimeError: If indexing fails
        """
        if not self.collection_exists():
            raise ValueError(
                f"Collection {self.settings.qdrant_collection_name} does not exist. "
                "Call create_collection() first."
            )

        count = 0
        batch: list[LegalArticle] = []
        batch_ids: set[int] = set()

        try:
            for article in articles:
                # Generate deterministic ID
                point_id = _generate_deterministic_id(_build_point_key(article))

                # Skip duplicates within the current batch
                if point_id in batch_ids:
                    logger.warning(
                        f"Duplicate ID detected for article {article.article_id}, skipping"
                    )
                    continue
                batch_ids.add(point_id)
                batch.append(article)

                # Batch insert
                if len(batch) >= self.settings.index_batch_size:
                    count += self._process_batch(batch)
                    logger.debug(
                        f"Indexed batch of {len(batch)} articles (total: {count})"
                    )
                    batch = []
                    batch_ids = set()

            # Insert remaining points
            if batch:
                count += self._process_batch(batch)
                logger.debug(f"Indexed final batch of {len(batch)} articles")

            logger.info(f"Successfully indexed {count} articles")
            return count

        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            raise RuntimeError(f"Indexing failed: {e}") from e

    def _process_batch(self, batch: list[LegalArticle]) -> int:
        texts = [article.content for article in batch]
        dense_embeddings = []
        sparse_embeddings = []

        for start in range(0, len(texts), self.settings.embedding_batch_size):
            chunk = texts[start : start + self.settings.embedding_batch_size]
            try:
                if self._use_native_api and self._flag_model is not None:
                    # Use BGEM3FlagModel native API for dense + sparse vectors
                    result = self._flag_model.encode(
                        chunk,
                        return_dense=True,
                        return_sparse=True,
                        return_colbert_vecs=False,
                    )
                    # Result is dict with 'dense_vecs' and 'lexical_weights'
                    if isinstance(result, dict):
                        dense_embeddings.extend(result.get("dense_vecs", []))
                        sparse_embeddings.extend(result.get("lexical_weights", []))
                    else:
                        dense_embeddings.extend(result)
                else:
                    # SentenceTransformer: dense-only encoding
                    chunk_embeddings = self.embedding_model.encode(
                        chunk, show_progress_bar=False
                    )
                    dense_embeddings.extend(chunk_embeddings)
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch chunk: {e}")
                raise RuntimeError(f"Embedding generation failed: {e}") from e

        # Build Qdrant points
        points = []
        for idx, article in enumerate(batch):
            point_id = _generate_deterministic_id(_build_point_key(article))
            dense_vector = dense_embeddings[idx].tolist() if idx < len(dense_embeddings) else None

            # Prepare sparse vector if available (only with native API)
            sparse_vector = None
            if self._has_sparse and idx < len(sparse_embeddings):
                sparse_vec = sparse_embeddings[idx]
                sparse_vector = self._convert_sparse_vector(sparse_vec, article.article_id)

            # Use named vectors for hybrid search, or default vector for dense-only
            if sparse_vector and self._has_sparse:
                vectors = {
                    "dense": dense_vector,
                    "sparse": sparse_vector,
                }
            else:
                vectors = dense_vector

            point = PointStruct(
                id=point_id,
                vector=vectors,
                payload={
                    "page_content": article.content,
                    "metadata": {
                        "article_id": article.article_id,
                        "title": article.title,
                        "law_id": article.law_id,
                        "law_title": article.law_title,
                        "law_short_name": article.law_short_name,
                        "chapter_id": article.chapter_id,
                        "chapter_title": article.chapter_title,
                        "source_anchor_id": article.source_anchor_id,
                        "doc_type": article.doc_type,
                        "editorial_notes": article.editorial_notes or [],
                        "cross_references": article.cross_references or [],
                        "url": article.url,
                    },
                },
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.settings.qdrant_collection_name,
            points=points,
        )
        return len(points)

    @staticmethod
    def _convert_sparse_vector(sparse_vec, article_id: str) -> dict | None:
        """Convert a sparse vector from various formats to Qdrant's {indices, values} format."""
        if isinstance(sparse_vec, dict):
            if "indices" in sparse_vec and "values" in sparse_vec:
                return {
                    "indices": list(sparse_vec["indices"]),
                    "values": list(sparse_vec["values"]),
                }
            # FlagEmbedding lexical_weights format: {token_id: weight, ...}
            indices = list(sparse_vec.keys())
            values = list(sparse_vec.values())
            if indices and values:
                return {
                    "indices": [int(i) for i in indices],
                    "values": [float(v) for v in values],
                }
        elif hasattr(sparse_vec, "indices") and hasattr(sparse_vec, "values"):
            # Scipy sparse format
            return {
                "indices": sparse_vec.indices.tolist() if hasattr(sparse_vec.indices, "tolist") else list(sparse_vec.indices),
                "values": sparse_vec.values.tolist() if hasattr(sparse_vec.values, "tolist") else list(sparse_vec.values),
            }
        elif isinstance(sparse_vec, (list, tuple)) and len(sparse_vec) == 2:
            return {
                "indices": list(sparse_vec[0]),
                "values": list(sparse_vec[1]),
            }
        logger.warning(f"Unrecognized sparse vector format for article {article_id}, skipping")
        return None
