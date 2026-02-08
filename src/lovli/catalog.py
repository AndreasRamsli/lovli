"""Law catalog generation for Tier 0 routing index."""

import json
import logging
from pathlib import Path
from typing import Iterator

from .parser import parse_law_header

logger = logging.getLogger(__name__)


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
    Generate a 2-3 sentence summary of a law using an LLM.

    Args:
        law_title: Full title of the law
        llm: LangChain LLM instance (ChatOpenAI or similar)

    Returns:
        Generated summary string
    """
    prompt = (
        f"Beskriv denne norske loven i 2-3 korte setninger på norsk: {law_title}. "
        f"Forklar hvem loven gjelder for og hva den regulerer. "
        f"Svar kun med beskrivelsen, ingen overskrift eller innledning."
    )

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        return content.strip()
    except Exception as e:
        logger.warning(f"Failed to generate summary for '{law_title}': {e}")
        return ""


def build_catalog(
    data_dir: Path,
    output_path: Path,
    llm=None,
    skip_summaries: bool = False,
) -> list[dict]:
    """
    Build a complete law catalog with optional LLM-generated summaries.

    Args:
        data_dir: Directory containing .xml law files
        output_path: Path to write the catalog JSON
        llm: Optional LLM instance for generating summaries
        skip_summaries: If True, skip LLM summary generation (headers only)

    Returns:
        List of catalog entries
    """
    catalog = []

    for idx, header in enumerate(scan_law_headers(data_dir)):
        entry = {**header, "summary": ""}

        if llm and not skip_summaries:
            logger.info(f"[{idx + 1}] Generating summary for: {header['law_title']}")
            entry["summary"] = generate_summary(header["law_title"], llm)
        else:
            logger.debug(f"[{idx + 1}] Scanned: {header['law_title']}")

        catalog.append(entry)

    # Write catalog to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    logger.info(f"Catalog written to {output_path} ({len(catalog)} entries)")
    return catalog


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
