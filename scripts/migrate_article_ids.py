#!/usr/bin/env python3
"""
Migrate indexed article IDs to canonical paragraph IDs.

This script builds an old->new mapping from source XML files, then scans a
Qdrant collection and rewrites metadata.article_id in-place (by re-upserting
the same point with updated payload).

Usage:
    # Dry run (default)
    python scripts/migrate_article_ids.py

    # Apply changes
    python scripts/migrate_article_ids.py --apply
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from lovli.config import get_settings  # noqa: E402
from lovli.parser import parse_xml_file  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_id_mapping(data_dirs: list[Path]) -> Dict[Tuple[str, str], str]:
    """
    Build mapping of (law_id, old_article_id) -> canonical_article_id.

    Uses parser output where `source_anchor_id` is the raw XML anchor and
    `article_id` is canonicalized.  Accepts multiple directories so that both
    nl/ (lover) and sf/ (forskrifter) can be processed in a single pass.
    """
    mapping: Dict[Tuple[str, str], str] = {}
    for data_dir in data_dirs:
        xml_files = sorted(data_dir.glob("*.xml"))
        logger.info("Scanning %s XML files in %s", len(xml_files), data_dir)
        for xml_path in xml_files:
            for article in parse_xml_file(xml_path):
                raw_id = article.source_anchor_id or article.article_id
                if raw_id != article.article_id:
                    mapping[(article.law_id, raw_id)] = article.article_id
    return mapping


def migrate_collection(
    client: QdrantClient, collection: str, mapping: Dict[Tuple[str, str], str], apply: bool
) -> tuple[int, int]:
    """Migrate Qdrant payload metadata.article_id values."""
    offset = None
    scanned = 0
    changed = 0

    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=128,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        updates = []
        for point in points:
            scanned += 1
            payload = dict(getattr(point, "payload", {}) or {})
            metadata = dict(payload.get("metadata", {}) or {})

            law_id = metadata.get("law_id")
            old_article_id = metadata.get("article_id")
            if not law_id or not old_article_id:
                continue

            new_article_id = mapping.get((law_id, old_article_id))
            if not new_article_id or new_article_id == old_article_id:
                continue

            metadata["article_id"] = new_article_id
            payload["metadata"] = metadata
            changed += 1

            if apply:
                updates.append(
                    PointStruct(
                        id=point.id,
                        vector=point.vector,
                        payload=payload,
                    )
                )

        if apply and updates:
            client.upsert(collection_name=collection, points=updates)

        if offset is None:
            break

    return scanned, changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate Qdrant article IDs to canonical IDs.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        action="append",
        dest="data_dirs",
        metavar="DIR",
        help=(
            "Directory containing law XML files. "
            "May be specified multiple times (e.g. --data-dir data/nl --data-dir data/sf). "
            "Defaults to data/nl and data/sf when not supplied."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply migration updates. Without this flag, script runs in dry-run mode.",
    )
    args = parser.parse_args()

    # Default: scan both nl/ and sf/ so no data directory is silently skipped.
    data_dirs: list[Path] = args.data_dirs or [ROOT_DIR / "data" / "nl", ROOT_DIR / "data" / "sf"]

    load_dotenv(ROOT_DIR / ".env")
    settings = get_settings()

    if settings.qdrant_in_memory:
        if settings.qdrant_persist_path:
            client = QdrantClient(path=settings.qdrant_persist_path)
        else:
            client = QdrantClient(":memory:")
    else:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    mapping = build_id_mapping(data_dirs)
    logger.info("Built %s old->new article ID mappings", len(mapping))

    scanned, changed = migrate_collection(
        client=client,
        collection=settings.qdrant_collection_name,
        mapping=mapping,
        apply=args.apply,
    )
    logger.info("Scanned %s points; %s would change", scanned, changed)
    if args.apply:
        logger.info(
            "Applied article ID migration to collection '%s'", settings.qdrant_collection_name
        )
    else:
        logger.info("Dry-run only. Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
