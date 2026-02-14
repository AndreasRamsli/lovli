#!/usr/bin/env python3
"""
Preflight duplicate check for parsed source IDs and computed point IDs.

Usage:
    # Check all laws
    python scripts/check_duplicate_point_ids.py

    # Check specific directory
    python scripts/check_duplicate_point_ids.py --data-dir data/nl
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from lovli.indexer import _build_point_key, _generate_deterministic_id  # noqa: E402
from lovli.parser import parse_xml_file  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check duplicate source IDs and point IDs before indexing.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT_DIR / "data" / "nl",
        help="Directory with Lovdata XML/HTML files (default: data/nl).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum duplicate examples to print per check.",
    )
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("*.xml"))
    if not files:
        logger.error("No input files found in %s", args.data_dir)
        sys.exit(1)

    source_keys: list[tuple[str, str]] = []
    point_ids: list[int] = []
    article_count = 0

    for file_path in files:
        for article in parse_xml_file(file_path):
            article_count += 1
            source_id = article.source_anchor_id or article.article_id
            source_keys.append((article.law_id, source_id))
            point_key = _build_point_key(article)
            point_ids.append(_generate_deterministic_id(point_key))

    source_counter = Counter(source_keys)
    point_counter = Counter(point_ids)

    duplicate_sources = [(k, c) for k, c in source_counter.items() if c > 1]
    duplicate_points = [(k, c) for k, c in point_counter.items() if c > 1]

    logger.info("Scanned %s files", len(files))
    logger.info("Parsed articles: %s", article_count)
    logger.info("Duplicate (law_id, source_id): %s", len(duplicate_sources))
    logger.info("Duplicate point IDs: %s", len(duplicate_points))

    if duplicate_sources:
        logger.info("Examples of duplicate source IDs:")
        for (law_id, source_id), count in duplicate_sources[: args.max_examples]:
            logger.info("  law_id=%s source_id=%s count=%s", law_id, source_id, count)

    if duplicate_points:
        logger.info("Examples of duplicate point IDs:")
        for point_id, count in duplicate_points[: args.max_examples]:
            logger.info("  point_id=%s count=%s", point_id, count)

    if duplicate_sources or duplicate_points:
        sys.exit(1)

    logger.info("No duplicates detected.")


if __name__ == "__main__":
    main()
