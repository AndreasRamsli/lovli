#!/usr/bin/env python3
"""
Build a law catalog (Tier 0) from Lovdata law files.

Scans all law files in a directory, extracts header metadata,
and optionally generates LLM summaries for each law.

Summary generation uses async concurrent LLM calls for speed.

Usage:
    # Headers only (fast, no LLM calls)
    python scripts/build_catalog.py data/nl/ --no-summaries

    # With LLM summaries (requires OPENROUTER_API_KEY)
    python scripts/build_catalog.py data/nl/

    # Higher concurrency for faster summary generation
    python scripts/build_catalog.py data/sf/ --concurrency 50

    # Retry failed/timed-out summaries from a previous run
    python scripts/build_catalog.py data/sf/ --backfill

    # Custom output path
    python scripts/build_catalog.py data/nl/ --output data/my_catalog.json
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

from lovli.catalog import build_catalog, backfill_summaries, DEFAULT_CONCURRENCY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _init_llm():
    """Initialize LLM for summary generation."""
    from lovli.config import get_settings
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.3,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        default_headers={
            "HTTP-Referer": "https://github.com/lovli",
            "X-Title": "Lovli Catalog Builder",
        },
    )
    logger.info(f"Using LLM for summaries: {settings.llm_model}")
    return llm


def main():
    parser = argparse.ArgumentParser(
        description="Build a law catalog from Lovdata files."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing .xml law files (e.g. data/nl/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root_dir / "data" / "law_catalog.json",
        help="Output path for the catalog JSON (default: data/law_catalog.json)",
    )
    parser.add_argument(
        "--no-summaries",
        action="store_true",
        help="Skip LLM summary generation (headers only, fast)",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Retry only missing summaries from an existing catalog (requires --output to exist)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent LLM requests for summary generation (default: {DEFAULT_CONCURRENCY})",
    )
    args = parser.parse_args()

    start_time = time.time()

    # Backfill mode: retry missing summaries from existing catalog
    if args.backfill:
        if not args.output.exists():
            logger.error(f"Catalog not found for backfill: {args.output}")
            sys.exit(1)
        try:
            llm = _init_llm()
        except Exception as e:
            logger.error(f"Could not initialize LLM for backfill: {e}")
            sys.exit(1)

        catalog = backfill_summaries(
            catalog_path=args.output,
            llm=llm,
            concurrency=args.concurrency,
        )
    else:
        # Normal build mode
        if not args.data_dir.is_dir():
            logger.error(f"Not a directory: {args.data_dir}")
            sys.exit(1)

        llm = None
        if not args.no_summaries:
            try:
                llm = _init_llm()
            except Exception as e:
                logger.warning(f"Could not initialize LLM, skipping summaries: {e}")
                llm = None

        catalog = build_catalog(
            data_dir=args.data_dir,
            output_path=args.output,
            llm=llm,
            skip_summaries=args.no_summaries or llm is None,
            concurrency=args.concurrency,
        )

    elapsed = time.time() - start_time

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Catalog build complete!")
    logger.info("=" * 60)
    logger.info(f"  Laws cataloged: {len(catalog)}")
    logger.info(f"  With summaries: {sum(1 for c in catalog if c.get('summary'))}")
    logger.info(f"  Missing summaries: {sum(1 for c in catalog if not c.get('summary'))}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
