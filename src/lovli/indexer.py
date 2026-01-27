"""Vector indexing module for storing legal articles in Qdrant."""

import hashlib
import logging
from typing import Iterator
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
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
            collections = self.client.get_collections()
            return name in [col.name for col in collections.collections]
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
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=self.settings.embedding_dimension,
                distance=Distance.COSINE,
            ),
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
        embeddings = []

        for start in range(0, len(texts), self.settings.embedding_batch_size):
            chunk = texts[start : start + self.settings.embedding_batch_size]
            try:
                chunk_embeddings = self.embedding_model.encode(
                    chunk, show_progress_bar=False
                )
                embeddings.extend(chunk_embeddings)
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch chunk: {e}")
                raise RuntimeError(f"Embedding generation failed: {e}") from e

        points = [
            PointStruct(
                id=_generate_deterministic_id(article.article_id),
                vector=embedding.tolist(),
                payload={
                    "article_id": article.article_id,
                    "title": article.title,
                    "content": article.content,
                    "law_id": article.law_id,
                    "law_title": article.law_title,
                    "url": article.url,
                },
            )
            for article, embedding in zip(batch, embeddings)
        ]

        self.client.upsert(
            collection_name=self.settings.qdrant_collection_name,
            points=points,
        )
        return len(points)
