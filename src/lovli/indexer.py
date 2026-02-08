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


def _generate_deterministic_id(article_id: str) -> int:
    """
    Generate a deterministic int64 ID from article ID string.

    Uses SHA-256 hash to avoid collisions while ensuring determinism.

    Args:
        article_id: Unique article identifier

    Returns:
        int64 ID suitable for Qdrant
    """
    # Use SHA-256 for deterministic hashing
    hash_bytes = hashlib.sha256(article_id.encode("utf-8")).digest()
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

        try:
            logger.info(f"Loading embedding model: {self.settings.embedding_model_name}")
            self.embedding_model = SentenceTransformer(
                self.settings.embedding_model_name
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise ValueError(f"Cannot load embedding model: {e}") from e

        # Detect BGE-M3 and native API availability once
        self._is_bge_m3 = "bge-m3" in self.settings.embedding_model_name.lower()
        self._use_native_api = False
        if self._is_bge_m3:
            try:
                from FlagEmbedding import FlagModel
                # Check if we can access the underlying FlagModel
                if hasattr(self.embedding_model, 'model') and isinstance(self.embedding_model.model, FlagModel):
                    self._use_native_api = True
                    logger.info("BGE-M3 native API detected and will be used")
            except ImportError:
                pass

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
        
        is_bge_m3 = "bge-m3" in self.settings.embedding_model_name.lower()
        
        if is_bge_m3:
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
        )
        logger.info(f"Collection {name} created successfully")

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
                point_id = _generate_deterministic_id(article.article_id)

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
                if self._use_native_api and self._is_bge_m3:
                    # Use BGE-M3's native API for sparse vectors
                    flag_model = self.embedding_model.model
                    result = flag_model.encode(
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
                elif self._is_bge_m3:
                    # Try sentence-transformers wrapper with return_sparse
                    # Note: This may not work with all sentence-transformers versions
                    try:
                        result = self.embedding_model.encode(
                            chunk,
                            return_sparse=True,
                            show_progress_bar=False
                        )
                        if isinstance(result, dict):
                            dense_embeddings.extend(result.get("dense", result.get("dense_vecs", [])))
                            sparse_embeddings.extend(result.get("sparse", result.get("lexical_weights", [])))
                        elif isinstance(result, tuple) and len(result) >= 2:
                            dense_embeddings.extend(result[0])
                            sparse_embeddings.extend(result[1])
                        else:
                            dense_embeddings.extend(result)
                    except (TypeError, AttributeError):
                        # Fallback to dense-only if sparse not supported
                        logger.warning("Sparse vectors not available, using dense-only")
                        chunk_embeddings = self.embedding_model.encode(
                            chunk, show_progress_bar=False
                        )
                        dense_embeddings.extend(chunk_embeddings)
                else:
                    # Standard dense-only encoding
                    chunk_embeddings = self.embedding_model.encode(
                        chunk, show_progress_bar=False
                    )
                    dense_embeddings.extend(chunk_embeddings)
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch chunk: {e}")
                raise RuntimeError(f"Embedding generation failed: {e}") from e

        # Store in QdrantVectorStore-compatible format:
        # - page_content: document text (for retrieval)
        # - metadata: nested dict with article info (for citations)
        points = []
        for idx, article in enumerate(batch):
            point_id = _generate_deterministic_id(article.article_id)
            dense_vector = dense_embeddings[idx].tolist() if idx < len(dense_embeddings) else None
            
            # Prepare sparse vector if available
            sparse_vector = None
            if is_bge_m3 and idx < len(sparse_embeddings):
                sparse_vec = sparse_embeddings[idx]
                # Convert sparse vector to Qdrant format: dict with indices and values
                if isinstance(sparse_vec, dict):
                    # Already in dict format (from FlagEmbedding)
                    if "indices" in sparse_vec and "values" in sparse_vec:
                        sparse_vector = {
                            "indices": list(sparse_vec["indices"]),
                            "values": list(sparse_vec["values"]),
                        }
                    elif "token_spans" in sparse_vec:
                        # Convert token_spans format to indices/values
                        indices = []
                        values = []
                        for token_id, weight in sparse_vec.get("token_spans", {}).items():
                            indices.append(int(token_id))
                            values.append(float(weight))
                        if indices and values and len(indices) == len(values):
                            sparse_vector = {"indices": indices, "values": values}
                        else:
                            logger.warning(f"Invalid token_spans format for article {article.article_id}, skipping sparse vector")
                elif hasattr(sparse_vec, 'indices') and hasattr(sparse_vec, 'values'):
                    # Scipy sparse format
                    sparse_vector = {
                        "indices": sparse_vec.indices.tolist() if hasattr(sparse_vec.indices, 'tolist') else list(sparse_vec.indices),
                        "values": sparse_vec.values.tolist() if hasattr(sparse_vec.values, 'tolist') else list(sparse_vec.values),
                    }
                elif isinstance(sparse_vec, (list, tuple)) and len(sparse_vec) == 2:
                    # Tuple/list format (indices, values)
                    sparse_vector = {
                        "indices": list(sparse_vec[0]),
                        "values": list(sparse_vec[1]),
                    }
            
            # Qdrant named vectors format:
            # - If sparse vectors are configured, use named vectors: {"dense": [...], "sparse": {...}}
            # - Otherwise, use default vector format (just the dense vector)
            # Note: QdrantVectorStore expects default vector, so we use named vectors only when sparse is available
            if sparse_vector and is_bge_m3:
                # Use named vectors format for hybrid search
                vectors = {
                    "dense": dense_vector,
                    "sparse": sparse_vector,
                }
            else:
                # Default vector format (dense-only, compatible with QdrantVectorStore)
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
