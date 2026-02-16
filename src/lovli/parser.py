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
_PARAGRAPH_REF_RE = re.compile(r"§\s*(\d+[A-Za-z]?)\s*-\s*(\d+\s*[A-Za-z]?)")
_EDITORIAL_NOTE_RE = re.compile(
    r"^\s*(?:\d+\s+)?(Endret ved (?:lov|forskrift)|Tilføyd ved (?:lov|forskrift)|Opphevet ved (?:lov|forskrift)|Tilføyet ved (?:lov|forskrift))",
    flags=re.IGNORECASE,
)
_EDITORIAL_CONTEXT_RE = re.compile(
    r"\b(endret ved|tilføyd ved|tilføyet ved|opphevet ved|ikr\.|i kraft)\b",
    flags=re.IGNORECASE,
)
_CANONICAL_LAW_REF_RE = re.compile(
    r"(lov|forskrift)/(\d{4}-\d{2}-\d{2})-(\d+)",
    flags=re.IGNORECASE,
)
_CHAPTER_PREFIX_RE = re.compile(r"^kapittel\s+\d+[a-zA-Z]?\.\s*", flags=re.IGNORECASE)


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
    source_anchor_id: str | None = None
    doc_type: str = "provision"
    linked_provision_id: str | None = None
    editorial_notes: list[dict] = field(default_factory=list)


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
    if short_elem:
        text = short_elem.get_text(strip=True)
        # Split on common separators: " – ", " - ", " — "
        for sep in (" – ", " — ", " - "):
            if sep in text:
                return text.split(sep)[0].strip()
        return text.strip() or None

    # Fallback for files without titleShort metadata.
    title_elem = soup.find("dd", class_="title") or soup.find("title")
    if not title_elem:
        return None
    title_text = title_elem.get_text(strip=True)
    # "Lov om ... (husleieloven)" -> "husleieloven"
    paren_match = re.search(r"\(([^)]+)\)", title_text)
    if paren_match:
        return paren_match.group(1).strip() or None
    return title_text.strip() or None


def _canonicalize_law_ref(value: str | None) -> str | None:
    """Return canonical law ref (lov/forskrift/YYYY-MM-DD-N) when possible."""
    if not value:
        return None
    match = _CANONICAL_LAW_REF_RE.search(value.strip())
    if not match:
        return None

    prefix, date_part, num_part = match.group(1).lower(), match.group(2), match.group(3)
    # Normalize leading zeros in law number (017 -> 17).
    normalized_num = str(int(num_part))
    return f"{prefix}/{date_part}-{normalized_num}"


def _extract_law_ref_from_soup(soup: BeautifulSoup) -> str | None:
    """Extract canonical law reference from in-file metadata."""
    dokid_elem = soup.find("dd", class_="dokid")
    if dokid_elem:
        canonical = _canonicalize_law_ref(dokid_elem.get_text(strip=True))
        if canonical:
            return canonical
    return None


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
    self_law_ref_normalized = _canonicalize_law_ref(self_law_ref)

    for a_tag in article_element.find_all("a", href=True):
        href = a_tag.get("href", "")
        if not href:
            continue

        # Only collect law and regulation references
        if not (href.startswith("lov/") or href.startswith("forskrift/")):
            continue

        # Exclude self-references (same law), even when numbering differs (e.g. 017 vs 17).
        href_law_ref = _canonicalize_law_ref(href)
        if self_law_ref_normalized and href_law_ref == self_law_ref_normalized:
            continue

        # Normalize: strip fragment identifiers for deduplication
        base_href = href.split("#")[0] if "#" in href else href

        dedupe_key = _canonicalize_law_ref(base_href) or base_href
        if dedupe_key not in seen:
            seen.add(dedupe_key)
            refs.append(base_href)

    return refs


def _canonicalize_article_id(
    raw_article_id: str,
    title_text: str,
    chapter_id: str | None,
) -> str:
    """
    Convert raw Lovdata anchor IDs to canonical paragraph IDs.

    Some Lovdata anchors can drift from visible paragraph numbering
    (e.g. around inserted sections like § 9-3 a). This function prefers
    the paragraph number shown in the article heading.
    """
    if not raw_article_id.startswith("kapittel-") or "-paragraf-" not in raw_article_id:
        return raw_article_id

    match = _PARAGRAPH_REF_RE.search(title_text or "")
    if not match:
        return raw_article_id

    chapter_ref, paragraph_ref = match.group(1), match.group(2)
    paragraph_ref = re.sub(r"\s+", "", paragraph_ref).lower()

    # Prefer chapter from heading when available, fall back to chapter_id/raw id.
    chapter_from_heading = chapter_ref.lower()
    if chapter_from_heading and chapter_from_heading != "0":
        chapter = chapter_from_heading
    elif chapter_id and chapter_id.startswith("kapittel-"):
        chapter = chapter_id.removeprefix("kapittel-").lower()
    else:
        chapter = raw_article_id.split("-")[1].lower()

    return f"kapittel-{chapter}-paragraf-{paragraph_ref}"


def _classify_doc_type(
    raw_article_id: str,
    title_text: str,
    article_content: str,
) -> str:
    """
    Classify parsed document chunks as substantive provisions or editorial notes.

    Editorial notes are usually amendment history snippets (e.g. "Endret ved lov...")
    that should stay attached to the same law, but be handled as supplemental context.
    """
    content = article_content or ""
    leading_content = content[:320]
    leading_content_lower = leading_content.lower()

    if _EDITORIAL_NOTE_RE.search(leading_content):
        return "editorial_note"

    if title_text and _EDITORIAL_CONTEXT_RE.search(title_text):
        return "editorial_note"

    # Fallback IDs indicate nodes without a stable Lovdata anchor in source.
    # Classify by content signals rather than forcing editorial for all fallback chunks.
    if "_art_" in raw_article_id:
        if _EDITORIAL_CONTEXT_RE.search(leading_content) and "skal lyde" not in leading_content_lower:
            return "editorial_note"

    return "provision"


def _is_nested_article_id(raw_article_id: str) -> bool:
    """Return True for fine-grained nested article IDs."""
    return "-ledd-" in raw_article_id or "-punkt-" in raw_article_id


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
    law_ref = _extract_law_ref_from_soup(soup) or _extract_law_ref_from_filename(law_id)

    # Find title - Lovdata uses <dd class="title">
    title_elem = soup.find("dd", class_="title")
    if not title_elem:
        # Fallback to <title> tag
        title_elem = soup.find("title")
    law_title_text = title_elem.get_text(strip=True) if title_elem else "Unknown Law"

    # Extract short name (e.g. "Husleieloven")
    law_short_name = _extract_short_name(soup)

    parsed_articles = _collect_articles(
        soup=soup,
        law_id=law_id,
        law_ref=law_ref,
        law_title_text=law_title_text,
        law_short_name=law_short_name,
        xml_path=xml_path,
        allow_nested_ids=False,
    )

    if not parsed_articles:
        logger.info("No coarse articles found in %s; retrying with nested IDs enabled", xml_path)
        parsed_articles = _collect_articles(
            soup=soup,
            law_id=law_id,
            law_ref=law_ref,
            law_title_text=law_title_text,
            law_short_name=law_short_name,
            xml_path=xml_path,
            allow_nested_ids=True,
        )

    if not parsed_articles:
        logger.warning(f"No articles found in {xml_path}")
        return

    yield from parsed_articles


def _collect_articles(
    soup: BeautifulSoup,
    law_id: str,
    law_ref: str,
    law_title_text: str,
    law_short_name: str | None,
    xml_path: Path,
    allow_nested_ids: bool,
) -> list[LegalArticle]:
    """Collect parsed articles from chapter sections and root-level articles."""
    sections = soup.find_all("section", id=re.compile(r"^kapittel-\d+[a-zA-Z]?$"))
    all_results: list[LegalArticle] = []
    seen_source_ids: set[str] = set()

    def add_unique(items: Iterator[LegalArticle]) -> None:
        for item in items:
            source_id = item.source_anchor_id or item.article_id
            if source_id in seen_source_ids:
                continue
            seen_source_ids.add(source_id)
            all_results.append(item)

    if sections:
        add_unique(
            _parse_hierarchical(
                sections=sections,
                law_id=law_id,
                law_ref=law_ref,
                law_title_text=law_title_text,
                law_short_name=law_short_name,
                xml_path=xml_path,
                allow_nested_ids=allow_nested_ids,
            )
        )

        section_article_ids = {id(article) for section in sections for article in section.find_all("article")}
        root_level_articles = [a for a in soup.find_all("article") if id(a) not in section_article_ids]
        if root_level_articles:
            add_unique(
                _parse_articles_flat(
                    articles=root_level_articles,
                    law_id=law_id,
                    law_ref=law_ref,
                    law_title_text=law_title_text,
                    law_short_name=law_short_name,
                    xml_path=xml_path,
                    allow_nested_ids=allow_nested_ids,
                )
            )
    else:
        articles = soup.find_all("article")
        if articles:
            add_unique(
                _parse_articles_flat(
                    articles=articles,
                    law_id=law_id,
                    law_ref=law_ref,
                    law_title_text=law_title_text,
                    law_short_name=law_short_name,
                    xml_path=xml_path,
                    allow_nested_ids=allow_nested_ids,
                )
            )

    return all_results


def _parse_hierarchical(
    sections: list[Tag],
    law_id: str,
    law_ref: str,
    law_title_text: str,
    law_short_name: str | None,
    xml_path: Path,
    allow_nested_ids: bool,
) -> Iterator[LegalArticle]:
    """Parse articles from chapter sections with hierarchical metadata."""
    total_count = 0

    for section in sections:
        last_provision_article_id: str | None = None
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
                chapter_id, chapter_title_clean, xml_path, allow_nested_ids,
            )
            if result:
                if result.doc_type == "editorial_note":
                    result.linked_provision_id = last_provision_article_id
                else:
                    last_provision_article_id = result.article_id
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
    allow_nested_ids: bool,
) -> Iterator[LegalArticle]:
    """Parse articles without chapter hierarchy (flat structure)."""
    last_provision_article_id: str | None = None
    for idx, article in enumerate(articles):
        result = _extract_article(
            article, idx, law_id, law_ref, law_title_text, law_short_name,
            chapter_id=None, chapter_title=None, xml_path=xml_path, allow_nested_ids=allow_nested_ids,
        )
        if result:
            if result.doc_type == "editorial_note":
                result.linked_provision_id = last_provision_article_id
            else:
                last_provision_article_id = result.article_id
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
    allow_nested_ids: bool = False,
) -> LegalArticle | None:
    """Extract a single LegalArticle from an <article> element."""
    try:
        # Article ID from the element's id attribute
        raw_article_id = article.get("id")
        if not raw_article_id:
            chapter_scope = chapter_id or "flat"
            # Include chapter scope to avoid collisions when idx resets per section.
            raw_article_id = f"{law_id}_{chapter_scope}_art_{idx}"

        # COARSE CHUNKING: Only index at paragraph level
        # Skip sub-articles like "kapittel-X-paragraf-Y-ledd-Z" or "X-punkt-Y"
        if not allow_nested_ids and _is_nested_article_id(raw_article_id):
            return None

        # Article title from <h3> element
        h3 = article.find("h3")
        title_text = h3.get_text(strip=True) if h3 else "Untitled Article"
        article_id = _canonicalize_article_id(raw_article_id, title_text, chapter_id)

        # Extract article content (full text including all sub-articles)
        article_content = article.get_text(separator="\n", strip=True)

        # Skip empty articles
        if not article_content.strip():
            return None

        doc_type = _classify_doc_type(raw_article_id, title_text, article_content)

        # Extract cross-references
        cross_refs = _extract_cross_references(article, law_ref)

        # Build Lovdata URL for this article using source anchor.
        url = f"{LOVDATA_BASE_URL}/{law_ref}#{raw_article_id}"

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
            source_anchor_id=raw_article_id,
            doc_type=doc_type,
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


def _build_editorial_note_payload(
    article: LegalArticle,
    editorial_note_max_chars: int | None = None,
) -> dict[str, str | None]:
    """Normalize an editorial note into a compact, deterministic payload shape."""
    content = (article.content or "").strip()
    if editorial_note_max_chars and editorial_note_max_chars > 0:
        content = content[:editorial_note_max_chars]
    return {
        "article_id": article.article_id,
        "content": content,
        "title": article.title,
        "source_anchor_id": article.source_anchor_id,
        "url": article.url,
        "chapter_id": article.chapter_id,
    }


def _sort_and_dedupe_editorial_payloads(
    notes: list[dict[str, str | None]],
    per_provision_cap: int | None = None,
) -> list[dict[str, str | None]]:
    """Sort and dedupe editorial note payloads deterministically."""
    ordered = sorted(
        notes,
        key=lambda note: (
            (note.get("source_anchor_id") or ""),
            (note.get("article_id") or ""),
            (note.get("content") or ""),
        ),
    )
    deduped: list[dict[str, str | None]] = []
    seen: set[tuple[str, str]] = set()
    for note in ordered:
        key = ((note.get("article_id") or "").strip(), (note.get("content") or "").strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(note)
    if per_provision_cap and per_provision_cap > 0:
        return deduped[:per_provision_cap]
    return deduped


def parse_xml_file_grouped(
    xml_path: Path,
    per_provision_cap: int | None = None,
    editorial_note_max_chars: int | None = None,
) -> Iterator[LegalArticle]:
    """
    Parse a Lovdata file and yield provisions with attached editorial notes.

    Editorial notes are grouped at parse-time and emitted under each provision
    as `editorial_notes` payload data. Standalone editorial-note articles are
    not yielded from this iterator.
    """
    articles = list(parse_xml_file(xml_path))
    if not articles:
        return

    provisions = [article for article in articles if article.doc_type == "provision"]
    if not provisions:
        return

    linked_notes: dict[tuple[str, str], list[dict[str, str | None]]] = {}
    chapter_fallback_notes: dict[tuple[str, str], list[dict[str, str | None]]] = {}
    chapter_provision_counts: dict[tuple[str, str], int] = {}

    for provision in provisions:
        chapter_key = ((provision.law_id or "").strip(), (provision.chapter_id or "").strip())
        if chapter_key[0] and chapter_key[1]:
            chapter_provision_counts[chapter_key] = chapter_provision_counts.get(chapter_key, 0) + 1

    for article in articles:
        if article.doc_type != "editorial_note":
            continue
        note_payload = _build_editorial_note_payload(
            article,
            editorial_note_max_chars=editorial_note_max_chars,
        )
        law_id = (article.law_id or "").strip()
        linked_provision_id = (article.linked_provision_id or "").strip()
        chapter_id = (article.chapter_id or "").strip()
        if law_id and linked_provision_id:
            linked_notes.setdefault((law_id, linked_provision_id), []).append(note_payload)
            continue
        if law_id and chapter_id:
            chapter_fallback_notes.setdefault((law_id, chapter_id), []).append(note_payload)

    for provision in provisions:
        law_id = (provision.law_id or "").strip()
        article_id = (provision.article_id or "").strip()
        chapter_id = (provision.chapter_id or "").strip()

        notes = list(linked_notes.get((law_id, article_id), []))
        chapter_key = (law_id, chapter_id)
        if (
            not notes
            and law_id
            and chapter_id
            and chapter_provision_counts.get(chapter_key, 0) == 1
        ):
            notes = list(chapter_fallback_notes.get(chapter_key, []))
        provision.editorial_notes = _sort_and_dedupe_editorial_payloads(
            notes,
            per_provision_cap=per_provision_cap,
        )
        yield provision


def parse_law_header(xml_path: Path) -> dict:
    """
    Parse only the header metadata from a Lovdata file (fast, no article extraction).

    Useful for building the law catalog without parsing all articles.

    Args:
        xml_path: Path to the Lovdata file

    Returns:
        Dictionary with law metadata: law_id, law_title, law_short_name,
        legal_area, date_in_force, chapter_count, article_count, chapter_titles
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"File not found: {xml_path}")

    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Use SoupStrainer to parse only the header for speed
    header_strainer = SoupStrainer("header")
    soup = BeautifulSoup(content, "html.parser", parse_only=header_strainer)

    law_id = xml_path.stem
    law_ref = _extract_law_ref_from_soup(soup) or _extract_law_ref_from_filename(law_id)

    title_elem = soup.find("dd", class_="title")
    if title_elem:
        law_title = title_elem.get_text(strip=True)
    else:
        title_match = re.search(r"<title>(.*?)</title>", content, flags=re.IGNORECASE | re.DOTALL)
        law_title = title_match.group(1).strip() if title_match else "Unknown Law"

    law_short_name = _extract_short_name(soup)

    # Legal area
    legal_area_elem = soup.find("dd", class_="legalArea")
    legal_area = legal_area_elem.get_text(strip=True) if legal_area_elem else None

    # Date in force
    date_elem = soup.find("dd", class_="dateInForce")
    date_in_force = date_elem.get_text(strip=True) if date_elem else None

    parsed_articles = list(parse_xml_file(xml_path))
    chapter_count = len({a.chapter_id for a in parsed_articles if a.chapter_id})
    article_count = len(parsed_articles)
    chapter_titles: list[str] = []
    chapter_titles_normalized: list[str] = []
    chapter_keywords: list[str] = []
    seen_chapter_ids: set[str] = set()
    seen_normalized_titles: set[str] = set()
    seen_keywords: set[str] = set()

    def _normalize_chapter_title(raw: str) -> str:
        cleaned = _CHAPTER_PREFIX_RE.sub("", (raw or "").strip())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _extract_chapter_keywords(text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]{4,}", (text or "").lower())
        deduped: list[str] = []
        seen_token: set[str] = set()
        for token in tokens:
            if token in seen_token:
                continue
            seen_token.add(token)
            deduped.append(token)
        return deduped[:8]

    for article in parsed_articles:
        chapter_id = (article.chapter_id or "").strip()
        chapter_title = (article.chapter_title or "").strip()
        if not chapter_id or not chapter_title:
            continue
        normalized_title = _normalize_chapter_title(chapter_title)
        if chapter_id in seen_chapter_ids:
            continue
        seen_chapter_ids.add(chapter_id)
        chapter_titles.append(chapter_title)
        if normalized_title and normalized_title not in seen_normalized_titles:
            seen_normalized_titles.add(normalized_title)
            chapter_titles_normalized.append(normalized_title)
            for keyword in _extract_chapter_keywords(normalized_title):
                if keyword in seen_keywords:
                    continue
                seen_keywords.add(keyword)
                chapter_keywords.append(keyword)

    return {
        "law_id": law_id,
        "law_ref": law_ref,
        "law_title": law_title,
        "law_short_name": law_short_name,
        "legal_area": legal_area,
        "date_in_force": date_in_force,
        "chapter_count": chapter_count,
        "article_count": article_count,
        "chapter_titles": chapter_titles,
        "chapter_titles_normalized": chapter_titles_normalized,
        "chapter_keywords": chapter_keywords,
    }
