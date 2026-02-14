#!/usr/bin/env python3
"""
CLI script for indexing Lovdata law files into Qdrant.

Usage:
    # Index a single file
    python scripts/index_laws.py data/nl/nl-19990326-017.xml

    # Index all files in a directory
    python scripts/index_laws.py data/nl/

    # Recreate the collection before indexing
    python scripts/index_laws.py data/nl/nl-19990326-017.xml --recreate

    # Use a custom collection name
    python scripts/index_laws.py data/nl/ --collection my_collection
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "src"))

from dotenv import load_dotenv
load_dotenv(root_dir / ".env")

from lovli.indexer import LegalIndexer
from lovli.parser import parse_xml_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def collect_files(path: Path) -> list[Path]:
    """Collect XML files from a path (file or directory)."""
    if path.is_file():
        return [path]
    elif path.is_dir():
        files = sorted(path.glob("*.xml"))
        if not files:
            logger.warning(f"No .xml files found in {path}")
        return files
    else:
        logger.error(f"Path does not exist: {path}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Index Lovdata law files into Qdrant vector database."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a single .xml file or a directory of .xml files",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the Qdrant collection before indexing",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Override the Qdrant collection name (default: from settings)",
    )
    args = parser.parse_args()

    # Collect files
    files = collect_files(args.path)
    if not files:
        logger.error("No files to index.")
        sys.exit(1)

    logger.info(f"Found {len(files)} file(s) to index")

    # Initialize indexer
    try:
        indexer = LegalIndexer()
    except Exception as e:
        logger.error(f"Failed to initialize indexer: {e}")
        sys.exit(1)

    # Apply collection name override
    if args.collection:
        indexer.settings.qdrant_collection_name = args.collection

    # Create or recreate collection
    collection_name = indexer.settings.qdrant_collection_name
    indexer.create_collection(collection_name=collection_name, recreate=args.recreate)
    indexer.ensure_payload_indexes(collection_name=collection_name)

    # Index each file
    total_articles = 0
    total_files = 0
    failed_files = []
    start_time = time.time()

    for file_path in files:
        try:
            logger.info(f"Parsing {file_path.name}...")
            count = indexer.index_articles(parse_xml_file(file_path))
            if count == 0:
                logger.warning(f"No articles indexed from {file_path.name}")
                continue

            total_articles += count
            total_files += 1
            logger.info(f"  Indexed {count} articles from {file_path.name} (total: {total_articles})")
        except Exception as e:
            logger.error(f"Failed to index {file_path.name}: {e}")
            failed_files.append(file_path.name)
            continue

    elapsed = time.time() - start_time

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Indexing complete!")
    logger.info("=" * 60)
    logger.info(f"  Files processed: {total_files}/{len(files)}")
    logger.info(f"  Articles indexed: {total_articles}")
    logger.info(f"  Collection: {collection_name}")
    logger.info(f"  Time: {elapsed:.1f}s")
    if failed_files:
        logger.warning(f"  Failed files ({len(failed_files)}): {', '.join(failed_files)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
