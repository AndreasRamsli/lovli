"""Parser for extracting legal articles from Lovdata HTML/XML files."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Base URL for Lovdata links
LOVDATA_BASE_URL = "https://lovdata.no"


@dataclass(slots=True)
class LegalArticle:
    """Represents a single legal article with metadata."""

    article_id: str
    title: str
    content: str
    law_id: str
    law_title: str
    url: str | None = None


def _extract_law_ref_from_filename(filename: str) -> str:
    """
    Extract Lovdata reference from filename.

    Example: nl-19990326-017 -> lov/1999-03-26-17
    """
    # Format: nl-YYYYMMDD-NNN
    parts = filename.split("-")
    if len(parts) >= 3:
        date_part = parts[1]  # YYYYMMDD
        num_part = parts[2]   # NNN
        if len(date_part) == 8:
            year = date_part[:4]
            month = date_part[4:6]
            day = date_part[6:8]
            return f"lov/{year}-{month}-{day}-{num_part}"
    return filename


def _parse_lovdata_html(xml_path: Path) -> Iterator[LegalArticle]:
    """
    Parse Lovdata HTML format (the actual format used in downloaded files).

    Lovdata files are HTML documents with:
    - Title in <dd class="title">
    - Articles in <article> elements with id like "kapittel-1-paragraf-1"
    - Article titles in <h3> elements
    """
    try:
        with open(xml_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError as e:
        logger.error(f"Failed to decode {xml_path}: {e}")
        raise ValueError(f"Cannot decode file as UTF-8: {xml_path}") from e
    except IOError as e:
        logger.error(f"Failed to read {xml_path}: {e}")
        raise IOError(f"Cannot read file: {xml_path}") from e

    if not content.strip():
        logger.warning(f"Empty file: {xml_path}")
        return

    try:
        # Use html.parser for Lovdata's HTML format
        soup = BeautifulSoup(content, "html.parser")
    except Exception as e:
        logger.error(f"Failed to parse HTML from {xml_path}: {e}")
        raise ValueError(f"Malformed file: {xml_path}") from e

    # Extract law metadata
    law_id = xml_path.stem
    law_ref = _extract_law_ref_from_filename(law_id)

    # Find title - Lovdata uses <dd class="title">
    title_elem = soup.find("dd", class_="title")
    if not title_elem:
        # Fallback to <title> tag
        title_elem = soup.find("title")
    law_title_text = title_elem.get_text(strip=True) if title_elem else "Unknown Law"

    # Find all articles - Lovdata uses <article> elements
    articles = soup.find_all("article")

    if not articles:
        logger.warning(f"No articles found in {xml_path}")
        return

    logger.info(f"Found {len(articles)} articles in {law_title_text}")

    for idx, article in enumerate(articles):
        try:
            # Article ID from the element's id attribute
            article_id = article.get("id") or f"{law_id}_art_{idx}"

            # COARSE CHUNKING: Only index at paragraph level
            # Skip sub-articles like "kapittel-X-paragraf-Y-ledd-Z" or "X-punkt-Y"
            # This ensures we index whole paragraphs for better semantic matching
            if "-ledd-" in article_id or "-punkt-" in article_id:
                logger.debug(f"Skipping sub-article {article_id} (indexing at paragraph level)")
                continue

            # Article title from <h3> element
            h3 = article.find("h3")
            title_text = h3.get_text(strip=True) if h3 else "Untitled Article"

            # Extract article content (full text including all sub-articles)
            article_content = article.get_text(separator="\n", strip=True)

            # Skip empty articles
            if not article_content.strip():
                logger.debug(f"Skipping empty article {article_id} in {xml_path}")
                continue

            # Build Lovdata URL for this article
            url = f"{LOVDATA_BASE_URL}/{law_ref}#{article_id}"

            yield LegalArticle(
                article_id=article_id,
                title=title_text,
                content=article_content,
                law_id=law_id,
                law_title=law_title_text,
                url=url,
            )
        except Exception as e:
            logger.error(f"Error processing article {idx} in {xml_path}: {e}")
            continue


def parse_xml_file(xml_path: Path) -> Iterator[LegalArticle]:
    """
    Parse a Lovdata file and extract legal articles.

    Lovdata files are HTML documents containing legal text with articles
    marked up using <article> elements.

    Args:
        xml_path: Path to the Lovdata file

    Yields:
        LegalArticle objects representing individual articles

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is malformed or empty
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"File not found: {xml_path}")

    if not xml_path.is_file():
        raise ValueError(f"Path is not a file: {xml_path}")

    yield from _parse_lovdata_html(xml_path)


def parse_law_directory(data_path: Path) -> Iterator[LegalArticle]:
    """
    Parse all XML files in a directory.

    Args:
        data_path: Path to directory containing XML files

    Yields:
        LegalArticle objects from all XML files

    Raises:
        ValueError: If directory doesn't exist or is not a directory
    """
    if not data_path.exists():
        raise ValueError(f"Directory does not exist: {data_path}")

    if not data_path.is_dir():
        raise ValueError(f"Path is not a directory: {data_path}")

    has_files = False
    for xml_file in data_path.glob("*.xml"):
        has_files = True
        try:
            yield from parse_xml_file(xml_file)
        except Exception as e:
            logger.error(f"Failed to parse {xml_file}: {e}")
            continue
    if not has_files:
        logger.warning(f"No XML files found in {data_path}")
