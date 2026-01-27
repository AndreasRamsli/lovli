#!/usr/bin/env python3
"""Index Husleieloven (tenant law) into Qdrant for testing."""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lovli.parser import parse_xml_file
from lovli.indexer import LegalIndexer
from lovli.config import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # Use in-memory Qdrant with persistence for testing
    # Note: openrouter_api_key is required but not used for indexing
    settings = Settings(
        openrouter_api_key="not-needed-for-indexing",
        qdrant_in_memory=True,
        qdrant_persist_path="./qdrant_data",
        embedding_model_name="BAAI/bge-m3",  # Multilingual model for Norwegian legal text
        embedding_dimension=1024,  # BGE-M3 dimension
        index_batch_size=50,
        embedding_batch_size=8,  # Smaller batch for larger model
    )

    # Path to Husleieloven
    husleie_path = Path("data/nl/nl-19990326-017.xml")
    
    if not husleie_path.exists():
        logger.error(f"File not found: {husleie_path}")
        sys.exit(1)

    logger.info("Initializing indexer...")
    indexer = LegalIndexer(settings)

    logger.info("Creating collection...")
    indexer.create_collection(recreate=True)

    logger.info(f"Parsing {husleie_path}...")
    articles = parse_xml_file(husleie_path)

    logger.info("Indexing articles...")
    count = indexer.index_articles(articles)

    logger.info(f"Successfully indexed {count} articles!")
    logger.info(f"Qdrant data persisted to: {settings.qdrant_persist_path}")


if __name__ == "__main__":
    main()
