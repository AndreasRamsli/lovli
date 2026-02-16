"""Law catalog generation for Tier 0 routing index."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Iterator

from .parser import parse_law_header

logger = logging.getLogger(__name__)

# Default concurrency limit for async LLM calls
DEFAULT_CONCURRENCY = 30

# Timeout per LLM request in seconds
_REQUEST_TIMEOUT = 30

# How often to log progress (every N completions)
_PROGRESS_INTERVAL = 100

_SUMMARY_PROMPT_TEMPLATE = (
    "Beskriv denne norske loven i 2-3 korte setninger på norsk: {law_title}. "
    "Forklar hvem loven gjelder for og hva den regulerer. "
    "Svar kun med beskrivelsen, ingen overskrift eller innledning."
)


def scan_law_headers(data_dir: Path) -> Iterator[dict]:
    """
    Scan all law files in a directory and extract header metadata.

    Args:
        data_dir: Directory containing .xml law files (e.g. data/nl/)

    Yields:
        Header metadata dictionaries from parse_law_header()
    """
    if not data_dir.is_dir():
        raise ValueError(f"Not a directory: {data_dir}")

    xml_files = sorted(data_dir.glob("*.xml"))
    logger.info(f"Scanning {len(xml_files)} law files in {data_dir}")

    for xml_path in xml_files:
        try:
            header = parse_law_header(xml_path)
            yield header
        except Exception as e:
            logger.warning(f"Failed to parse header from {xml_path.name}: {e}")
            continue


def generate_summary(law_title: str, llm) -> str:
    """
    Generate a 2-3 sentence summary of a law using an LLM (synchronous).

    Args:
        law_title: Full title of the law
        llm: LangChain LLM instance (ChatOpenAI or similar)

    Returns:
        Generated summary string
    """
    prompt = _SUMMARY_PROMPT_TEMPLATE.format(law_title=law_title)

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        return content.strip()
    except Exception as e:
        logger.warning(f"Failed to generate summary for '{law_title}': {e}")
        return ""


async def _generate_summary_async(
    idx: int,
    law_title: str,
    llm,
    semaphore: asyncio.Semaphore,
    progress: dict,
) -> str:
    """
    Generate a summary for a single law asynchronously with concurrency control.

    Args:
        idx: Entry index (for logging)
        law_title: Full title of the law
        llm: LangChain LLM instance with async support (ChatOpenAI)
        semaphore: Semaphore to limit concurrent requests
        progress: Shared dict tracking completion count and total

    Returns:
        Generated summary string, or empty string on failure
    """
    prompt = _SUMMARY_PROMPT_TEMPLATE.format(law_title=law_title)

    async with semaphore:
        try:
            response = await asyncio.wait_for(
                llm.ainvoke(prompt),
                timeout=_REQUEST_TIMEOUT,
            )
            content = response.content if hasattr(response, "content") else str(response)
            summary = content.strip()
        except asyncio.TimeoutError:
            logger.warning(f"[{idx + 1}] Timeout after {_REQUEST_TIMEOUT}s for: {law_title}")
            summary = ""
        except Exception as e:
            logger.warning(f"[{idx + 1}] Failed for '{law_title}': {e}")
            summary = ""

    # Update and log progress
    progress["done"] += 1
    done = progress["done"]
    total = progress["total"]
    if done % _PROGRESS_INTERVAL == 0 or done == total:
        elapsed = time.time() - progress["start_time"]
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        logger.info(f"  Progress: {done}/{total} ({done * 100 // total}%) - {rate:.1f}/s - ETA {eta:.0f}s")

    return summary


async def _generate_summaries_batch(
    catalog: list[dict],
    llm,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[dict]:
    """
    Generate summaries for all catalog entries concurrently.

    Args:
        catalog: List of catalog entries (modified in-place)
        llm: LangChain LLM instance with async support
        concurrency: Max number of concurrent LLM requests

    Returns:
        The catalog list with summaries populated
    """
    semaphore = asyncio.Semaphore(concurrency)
    total = len(catalog)
    progress = {"done": 0, "total": total, "start_time": time.time()}
    logger.info(f"Generating summaries for {total} entries (concurrency={concurrency}, timeout={_REQUEST_TIMEOUT}s)")

    tasks = [
        _generate_summary_async(idx, entry["law_title"], llm, semaphore, progress)
        for idx, entry in enumerate(catalog)
    ]

    summaries = await asyncio.gather(*tasks)

    for entry, summary in zip(catalog, summaries):
        entry["summary"] = summary

    succeeded = sum(1 for s in summaries if s)
    elapsed = time.time() - progress["start_time"]
    logger.info(f"Summary generation complete: {succeeded}/{total} succeeded in {elapsed:.1f}s")
    return catalog


def build_catalog(
    data_dir: Path,
    output_path: Path,
    llm=None,
    skip_summaries: bool = False,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[dict]:
    """
    Build a complete law catalog with optional LLM-generated summaries.

    When summaries are enabled, uses async concurrent LLM calls for speed.

    Args:
        data_dir: Directory containing .xml law files
        output_path: Path to write the catalog JSON
        llm: Optional LLM instance for generating summaries
        skip_summaries: If True, skip LLM summary generation (headers only)
        concurrency: Max number of concurrent LLM requests (default: 30)

    Returns:
        List of catalog entries
    """
    # Phase 1: Scan all headers (fast, CPU-bound)
    catalog = [{**header, "summary": ""} for header in scan_law_headers(data_dir)]
    logger.info(f"Scanned {len(catalog)} law headers")

    # Phase 2: Generate summaries concurrently (slow, I/O-bound)
    if llm and not skip_summaries:
        asyncio.run(_generate_summaries_batch(catalog, llm, concurrency))

    # Phase 3: Write output
    _write_catalog(catalog, output_path)
    return catalog


def build_catalog_multi(
    data_dirs: list[Path],
    output_path: Path,
    llm=None,
    skip_summaries: bool = False,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[dict]:
    """
    Build a merged catalog from multiple law directories.

    Entries are deduplicated by `law_id` with first-seen precedence to keep
    output deterministic.
    """
    if not data_dirs:
        raise ValueError("At least one data directory is required.")

    catalog: list[dict] = []
    seen_law_ids: set[str] = set()
    for data_dir in data_dirs:
        logger.info("Scanning catalog input directory: %s", data_dir)
        for header in scan_law_headers(data_dir):
            law_id = (header.get("law_id") or "").strip()
            if not law_id or law_id in seen_law_ids:
                continue
            seen_law_ids.add(law_id)
            catalog.append({**header, "summary": ""})

    logger.info("Scanned %s unique law headers from %s directories", len(catalog), len(data_dirs))

    if llm and not skip_summaries:
        asyncio.run(_generate_summaries_batch(catalog, llm, concurrency))

    _write_catalog(catalog, output_path)
    return catalog


def backfill_summaries(
    catalog_path: Path,
    llm,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[dict]:
    """
    Load an existing catalog and generate summaries only for entries missing one.

    Useful for retrying after timeouts or failures without redoing all work.

    Args:
        catalog_path: Path to the existing catalog JSON
        llm: LangChain LLM instance for generating summaries
        concurrency: Max number of concurrent LLM requests

    Returns:
        Updated catalog list
    """
    catalog = load_catalog(catalog_path)
    total = len(catalog)
    missing = [i for i, entry in enumerate(catalog) if not entry.get("summary")]

    if not missing:
        logger.info(f"All {total} entries already have summaries, nothing to backfill")
        return catalog

    logger.info(f"Backfilling {len(missing)}/{total} missing summaries")

    # Build a subset catalog for the missing entries
    missing_entries = [catalog[i] for i in missing]
    asyncio.run(_generate_summaries_batch(missing_entries, llm, concurrency))

    # Write back to the same file
    for idx, entry in zip(missing, missing_entries):
        catalog[idx]["summary"] = entry["summary"]

    _write_catalog(catalog, catalog_path)

    filled = sum(1 for i in missing if catalog[i].get("summary"))
    still_missing = len(missing) - filled
    logger.info(f"Backfill complete: {filled}/{len(missing)} filled, {still_missing} still missing")
    return catalog


def _write_catalog(catalog: list[dict], output_path: Path) -> None:
    """Write catalog to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    logger.info(f"Catalog written to {output_path} ({len(catalog)} entries)")


def load_catalog(catalog_path: Path) -> list[dict]:
    """
    Load a previously built law catalog from JSON.

    Args:
        catalog_path: Path to the catalog JSON file

    Returns:
        List of catalog entries
    """
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    with open(catalog_path, "r", encoding="utf-8") as f:
        return json.load(f)
