#!/usr/bin/env python3
"""
Validate post-reindex metadata completeness and retrieval behavior.

Checks:
- metadata.doc_type exists on all indexed points
- doc_type distribution sanity (provision/editorial_note)
- retrieval smoke checks keep provisions ahead of editorial notes

Usage:
    # Validate current configured collection
    python scripts/validate_reindex.py

    # Validate a specific collection and fail on missing doc_type
    python scripts/validate_reindex.py --collection lovli_laws --require-zero-missing

    # Include retrieval smoke checks (requires full runtime deps and credentials)
    python scripts/validate_reindex.py --with-smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from lovli.chain import LegalRAGChain  # noqa: E402
from lovli.config import get_settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)


SMOKE_QUERIES = [
    "Hva sier husleieloven om oppsigelsesfrist?",
    "Når ble denne bestemmelsen endret?",
    "Hva kan utleier kreve i depositum?",
]


def create_client() -> QdrantClient:
    """Create Qdrant client from app settings."""
    settings = get_settings()
    if settings.qdrant_in_memory:
        if settings.qdrant_persist_path:
            return QdrantClient(path=settings.qdrant_persist_path)
        return QdrantClient(":memory:")
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def scan_doc_type_metrics(client: QdrantClient, collection_name: str) -> dict[str, int]:
    """Scan full collection for doc_type completeness and distribution."""
    offset = None
    total_points = 0
    missing_doc_type = 0
    provision_count = 0
    editorial_note_count = 0
    other_doc_type_count = 0

    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        for point in points:
            total_points += 1
            payload = getattr(point, "payload", {}) or {}
            metadata = payload.get("metadata", {}) or {}
            doc_type = (metadata.get("doc_type") or "").strip()
            if not doc_type:
                missing_doc_type += 1
            elif doc_type == "provision":
                provision_count += 1
            elif doc_type == "editorial_note":
                editorial_note_count += 1
            else:
                other_doc_type_count += 1

        if offset is None:
            break

    return {
        "total_points": total_points,
        "missing_doc_type": missing_doc_type,
        "provision_count": provision_count,
        "editorial_note_count": editorial_note_count,
        "other_doc_type_count": other_doc_type_count,
    }


def run_retrieval_smoke_checks() -> list[dict[str, object]]:
    """Run lightweight retrieval checks against the current collection."""
    chain = LegalRAGChain()
    results: list[dict[str, object]] = []
    for query in SMOKE_QUERIES:
        sources, _top_score, _scores = chain.retrieve(query)
        doc_types = [s.get("doc_type") for s in sources]
        first_editorial_idx = next(
            (idx for idx, doc_type in enumerate(doc_types) if doc_type == "editorial_note"),
            None,
        )
        provision_after_editorial = False
        if first_editorial_idx is not None:
            provision_after_editorial = any(
                doc_type == "provision" for doc_type in doc_types[first_editorial_idx + 1 :]
            )
        results.append(
            {
                "query": query,
                "sources_count": len(sources),
                "doc_types": doc_types,
                "provision_after_editorial": provision_after_editorial,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate doc_type completeness and retrieval quality.")
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection name override (default: from settings).",
    )
    parser.add_argument(
        "--with-smoke",
        action="store_true",
        help="Run retrieval smoke checks (off by default).",
    )
    parser.add_argument(
        "--require-zero-missing",
        action="store_true",
        help="Exit with non-zero status if any points are missing metadata.doc_type.",
    )
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env")
    settings = get_settings()
    collection_name = args.collection or settings.qdrant_collection_name

    client = create_client()
    metrics = scan_doc_type_metrics(client, collection_name)
    logger.info("Collection: %s", collection_name)
    logger.info("total_points=%s", metrics["total_points"])
    logger.info("missing_doc_type=%s", metrics["missing_doc_type"])
    logger.info("provision_count=%s", metrics["provision_count"])
    logger.info("editorial_note_count=%s", metrics["editorial_note_count"])
    logger.info("other_doc_type_count=%s", metrics["other_doc_type_count"])

    if args.with_smoke:
        smoke_results = run_retrieval_smoke_checks()
        for result in smoke_results:
            logger.info(
                "smoke query='%s' sources=%s doc_types=%s provision_after_editorial=%s",
                result["query"],
                result["sources_count"],
                result["doc_types"],
                result["provision_after_editorial"],
            )
        violations = [r for r in smoke_results if r["provision_after_editorial"]]
        if violations:
            logger.warning(
                "Detected %s ordering violations where a provision appears after an editorial note.",
                len(violations),
            )
    else:
        logger.info("Smoke checks skipped (use --with-smoke to enable).")

    if args.require_zero_missing and metrics["missing_doc_type"] > 0:
        logger.error("Validation failed: missing_doc_type=%s", metrics["missing_doc_type"])
        sys.exit(1)

    logger.info("Validation completed.")


if __name__ == "__main__":
    main()
