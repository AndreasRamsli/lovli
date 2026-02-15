#!/usr/bin/env python3
"""Create required Qdrant payload indexes for filtering."""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
load_dotenv(ROOT_DIR / ".env")

from lovli.indexer import LegalIndexer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensure Qdrant payload indexes for metadata.law_id/chapter_id/doc_type."
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Override collection name (default: from settings)",
    )
    args = parser.parse_args()

    indexer = LegalIndexer()
    collection = args.collection or indexer.settings.qdrant_collection_name
    logger.info("Ensuring payload indexes for collection: %s", collection)
    indexer.ensure_payload_indexes(collection_name=collection)
    logger.info("Done.")


if __name__ == "__main__":
    main()

