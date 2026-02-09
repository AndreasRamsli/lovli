"""Parser for extracting legal articles from Lovdata HTML/XML files."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup, Tag, SoupStrainer

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
    law_short_name: str | None = None
    chapter_id: str | None = None
    chapter_title: str | None = None
    cross_references: list[str] = field(default_factory=list)
    url: str | None = None


def _extract_law_ref_from_filename(filename: str) -> str:
    """
    Extract Lovdata reference from filename.

    Examples:
        nl-19990326-017 -> lov/1999-03-26-017
        sf-19970606-032 -> forskrift/1997-06-06-032
    """
    # Format: nl-YYYYMMDD-NNN or sf-YYYYMMDD-NNN
    parts = filename.split("-")
    if len(parts) >= 3:
        prefix = parts[0]  # "nl" or "sf"
        date_part = parts[1]  # YYYYMMDD
        num_part = parts[2]   # NNN
        if len(date_part) == 8:
            year = date_part[:4]
            month = date_part[4:6]
            day = date_part[6:8]
            # Map prefix to Lovdata URL prefix
            if prefix == "sf":
                ref_prefix = "forskrift"
            elif prefix == "nl":
                ref_prefix = "lov"
            else:
                # Unknown prefix, default to lov
                ref_prefix = "lov"
            return f"{ref_prefix}/{year}-{month}-{day}-{num_part}"
    return filename


def _extract_short_name(soup: BeautifulSoup) -> str | None:
    """
    Extract law short name from header metadata.

    <dd class="titleShort">Husleieloven – husll</dd>
    -> "Husleieloven"
    """
    short_elem = soup.find("dd", class_="titleShort")
    if not short_elem:
        return None
    text = short_elem.get_text(strip=True)
    # Split on common separators: " – ", " - ", " — "
    for sep in (" – ", " — ", " - "):
        if sep in text:
            return text.split(sep)[0].strip()
    return text.strip() or None


def _extract_cross_references(article_element: Tag, self_law_ref: str) -> list[str]:
    """
    Extract cross-references from an article element.

    Looks for <a href="lov/..."> and <a href="forskrift/..."> links
    and returns deduplicated list of external references.

    Args:
        article_element: BeautifulSoup article Tag
        self_law_ref: The law reference for the current file (to exclude self-refs)

    Returns:
        Deduplicated list of reference hrefs (e.g. ["lov/2002-06-21-34", "lov/2005-06-17-90"])
    """
    refs = []
    seen = set()

    for a_tag in article_element.find_all("a", href=True):
        href = a_tag.get("href", "")
        if not href:
            continue

        # Only collect law and regulation references
        if not (href.startswith("lov/") or href.startswith("forskrift/")):
            continue

        # Exclude self-references (same law)
        if self_law_ref and href.startswith(self_law_ref):
            continue

        # Normalize: strip fragment identifiers for deduplication
        base_href = href.split("#")[0] if "#" in href else href

        if base_href not in seen:
            seen.add(base_href)
            refs.append(base_href)

    return refs


def _parse_lovdata_html(xml_path: Path) -> Iterator[LegalArticle]:
    """
    Parse Lovdata HTML format with hierarchical extraction.

    Lovdata files are HTML documents with:
    - Title in <dd class="title">
    - Short name in <dd class="titleShort">
    - Chapters in <section id="kapittel-X"> with <h2> headings
    - Articles in <article> elements with id like "kapittel-1-paragraf-1"
    - Article titles in <h3> elements
    - Cross-references as <a href="lov/..."> links
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

    # Extract short name (e.g. "Husleieloven")
    law_short_name = _extract_short_name(soup)

    # Try hierarchical extraction via <section> elements (chapters)
    sections = soup.find_all("section", id=re.compile(r"^kapittel-\d+[a-zA-Z]?$"))

    if sections:
        # Hierarchical path: iterate sections -> articles
        yield from _parse_hierarchical(
            sections, law_id, law_ref, law_title_text, law_short_name, xml_path
        )
    else:
        # Flat fallback: iterate all articles directly (for laws without chapter structure)
        articles = soup.find_all("article")
        if not articles:
            logger.warning(f"No articles found in {xml_path}")
            return

        logger.info(f"Found {len(articles)} articles in {law_title_text} (flat structure)")
        yield from _parse_articles_flat(
            articles, law_id, law_ref, law_title_text, law_short_name, xml_path
        )


def _parse_hierarchical(
    sections: list[Tag],
    law_id: str,
    law_ref: str,
    law_title_text: str,
    law_short_name: str | None,
    xml_path: Path,
) -> Iterator[LegalArticle]:
    """Parse articles from chapter sections with hierarchical metadata."""
    total_count = 0

    for section in sections:
        chapter_id = section.get("id", "")
        h2 = section.find("h2")
        chapter_title = h2.get_text(strip=True) if h2 else None

        # Clean chapter title: remove "Kapittel X. " prefix to get just the name
        chapter_title_clean = chapter_title
        if chapter_title:
            match = re.match(r"^Kapittel\s+\d+[A-Za-z]?\.\s*", chapter_title)
            if match:
                chapter_title_clean = chapter_title[match.end():].strip() or chapter_title

        articles = section.find_all("article")
        for idx, article in enumerate(articles):
            result = _extract_article(
                article, idx, law_id, law_ref, law_title_text, law_short_name,
                chapter_id, chapter_title_clean, xml_path,
            )
            if result:
                total_count += 1
                yield result

    logger.info(f"Found {total_count} articles in {law_title_text} ({len(sections)} chapters)")


def _parse_articles_flat(
    articles: list[Tag],
    law_id: str,
    law_ref: str,
    law_title_text: str,
    law_short_name: str | None,
    xml_path: Path,
) -> Iterator[LegalArticle]:
    """Parse articles without chapter hierarchy (flat structure)."""
    for idx, article in enumerate(articles):
        result = _extract_article(
            article, idx, law_id, law_ref, law_title_text, law_short_name,
            chapter_id=None, chapter_title=None, xml_path=xml_path,
        )
        if result:
            yield result


def _extract_article(
    article: Tag,
    idx: int,
    law_id: str,
    law_ref: str,
    law_title_text: str,
    law_short_name: str | None,
    chapter_id: str | None,
    chapter_title: str | None,
    xml_path: Path,
) -> LegalArticle | None:
    """Extract a single LegalArticle from an <article> element."""
    try:
        # Article ID from the element's id attribute
        article_id = article.get("id") or f"{law_id}_art_{idx}"

        # COARSE CHUNKING: Only index at paragraph level
        # Skip sub-articles like "kapittel-X-paragraf-Y-ledd-Z" or "X-punkt-Y"
        if "-ledd-" in article_id or "-punkt-" in article_id:
            return None

        # Article title from <h3> element
        h3 = article.find("h3")
        title_text = h3.get_text(strip=True) if h3 else "Untitled Article"

        # Extract article content (full text including all sub-articles)
        article_content = article.get_text(separator="\n", strip=True)

        # Skip empty articles
        if not article_content.strip():
            return None

        # Extract cross-references
        cross_refs = _extract_cross_references(article, law_ref)

        # Build Lovdata URL for this article
        url = f"{LOVDATA_BASE_URL}/{law_ref}#{article_id}"

        return LegalArticle(
            article_id=article_id,
            title=title_text,
            content=article_content,
            law_id=law_id,
            law_title=law_title_text,
            law_short_name=law_short_name,
            chapter_id=chapter_id,
            chapter_title=chapter_title,
            cross_references=cross_refs,
            url=url,
        )
    except Exception as e:
        logger.error(f"Error processing article {idx} in {xml_path}: {e}")
        return None


def parse_xml_file(xml_path: Path) -> Iterator[LegalArticle]:
    """
    Parse a Lovdata file and extract legal articles.

    Lovdata files are HTML documents containing legal text with articles
    marked up using <article> elements, organized into chapters via
    <section> elements.

    Args:
        xml_path: Path to the Lovdata file

    Yields:
        LegalArticle objects representing individual articles with
        hierarchical metadata (chapter, cross-references, etc.)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is malformed or empty
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"File not found: {xml_path}")

    if not xml_path.is_file():
        raise ValueError(f"Path is not a file: {xml_path}")

    yield from _parse_lovdata_html(xml_path)


def parse_law_header(xml_path: Path) -> dict:
    """
    Parse only the header metadata from a Lovdata file (fast, no article extraction).

    Useful for building the law catalog without parsing all articles.

    Args:
        xml_path: Path to the Lovdata file

    Returns:
        Dictionary with law metadata: law_id, law_title, law_short_name,
        legal_area, date_in_force, chapter_count, article_count
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"File not found: {xml_path}")

    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Use SoupStrainer to parse only the header for speed
    header_strainer = SoupStrainer("header")
    soup = BeautifulSoup(content, "html.parser", parse_only=header_strainer)

    law_id = xml_path.stem
    law_ref = _extract_law_ref_from_filename(law_id)

    title_elem = soup.find("dd", class_="title")
    law_title = title_elem.get_text(strip=True) if title_elem else "Unknown Law"

    law_short_name = _extract_short_name(soup)

    # Legal area
    legal_area_elem = soup.find("dd", class_="legalArea")
    legal_area = legal_area_elem.get_text(strip=True) if legal_area_elem else None

    # Date in force
    date_elem = soup.find("dd", class_="dateInForce")
    date_in_force = date_elem.get_text(strip=True) if date_elem else None

    # Count chapters and articles using fast regex on raw content
    chapter_count = len(re.findall(r'<section[^>]+id="kapittel-\d+[a-zA-Z]?"', content))
    
    # Count articles: match all <article> tags and extract IDs, excluding sub-articles
    # This matches both hierarchical (kapittel-X-paragraf-Y) and flat structure articles
    # Exclude -ledd- and -punkt- sub-articles (same logic as _extract_article)
    all_article_matches = re.findall(r'<article[^>]+id="([^"]+)"', content)
    article_count = sum(
        1 for art_id in all_article_matches
        if "-ledd-" not in art_id and "-punkt-" not in art_id
    )

    return {
        "law_id": law_id,
        "law_ref": law_ref,
        "law_title": law_title,
        "law_short_name": law_short_name,
        "legal_area": legal_area,
        "date_in_force": date_in_force,
        "chapter_count": chapter_count,
        "article_count": article_count,
    }
