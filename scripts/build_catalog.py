#!/usr/bin/env python3
"""
Build a law catalog (Tier 0) from Lovdata law files.

Scans all law files in a directory, extracts header metadata,
and optionally generates LLM summaries for each law.

Usage:
    # Headers only (fast, no LLM calls)
    python scripts/build_catalog.py data/nl/ --no-summaries

    # With LLM summaries (requires OPENROUTER_API_KEY)
    python scripts/build_catalog.py data/nl/

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

from lovli.catalog import build_catalog

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        logger.error(f"Not a directory: {args.data_dir}")
        sys.exit(1)

    # Initialize LLM if summaries are requested
    llm = None
    if not args.no_summaries:
        try:
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
        except Exception as e:
            logger.warning(f"Could not initialize LLM, skipping summaries: {e}")
            llm = None

    start_time = time.time()
    catalog = build_catalog(
        data_dir=args.data_dir,
        output_path=args.output,
        llm=llm,
        skip_summaries=args.no_summaries or llm is None,
    )
    elapsed = time.time() - start_time

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Catalog build complete!")
    logger.info("=" * 60)
    logger.info(f"  Laws cataloged: {len(catalog)}")
    logger.info(f"  With summaries: {sum(1 for c in catalog if c.get('summary'))}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
