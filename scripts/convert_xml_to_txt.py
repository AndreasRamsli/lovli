#!/usr/bin/env python3
from __future__ import annotations

"""Convert Lovdata XML/HTML files to clean markdown-formatted .txt files.

Produces one .txt file per source XML, suitable for ingestion into RAGFlow
or other knowledge-base systems.

Usage:
    # Single file (development):
    python scripts/convert_xml_to_txt.py --input-file data/nl/nl-19990326-017.xml

    # Batch — all files in a directory:
    python scripts/convert_xml_to_txt.py --input-dir data/nl --output-dir data/txt/nl
    python scripts/convert_xml_to_txt.py --input-dir data/sf --output-dir data/txt/sf
"""

import argparse
import logging
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

logger = logging.getLogger(__name__)

# Article classes that contain substantive legal text and should be rendered
# like a legalArticle (same internal structure: legalP, listArticle children).
_LEGAL_ARTICLE_CLASSES = {"legalArticle", "futureLegalArticle"}

# ---------------------------------------------------------------------------
# Metadata helpers (adapted from src/lovli/parser.py)
# ---------------------------------------------------------------------------


def _extract_law_ref_from_filename(filename: str) -> str:
    """nl-19990326-017 -> LOV-1999-03-26-17, sf-… -> FOR-…"""
    parts = filename.split("-")
    if len(parts) >= 3 and len(parts[1]) == 8:
        prefix = "LOV" if parts[0] == "nl" else "FOR"
        year, month, day = parts[1][:4], parts[1][4:6], parts[1][6:8]
        num = str(int(parts[2]))
        return f"{prefix}-{year}-{month}-{day}-{num}"
    return filename


def _text(elem) -> str:
    """Get stripped text from an element, or empty string."""
    if elem is None:
        return ""
    return elem.get_text(strip=True)


def _extract_metadata(soup: BeautifulSoup, filename: str) -> dict[str, str]:
    """Extract header metadata into a dict."""
    meta: dict[str, str] = {}

    title_elem = soup.find("dd", class_="title")
    if title_elem:
        meta["Tittel"] = _text(title_elem)
    else:
        t = soup.find("title")
        if t:
            meta["Tittel"] = _text(t)

    short = soup.find("dd", class_="titleShort")
    if short:
        raw = _text(short)
        for sep in (" \u2013 ", " \u2014 ", " - "):
            if sep in raw:
                raw = raw.split(sep)[0].strip()
                break
        if raw:
            meta["Korttittel"] = raw

    meta["Dokument-ID"] = _extract_law_ref_from_filename(filename)

    dept = soup.find("dd", class_="ministry")
    if dept:
        meta["Departement"] = _text(dept)

    date_in_force = soup.find("dd", class_="dateInForce")
    if date_in_force:
        meta["I kraft fra"] = _text(date_in_force)

    last_change = soup.find("dd", class_="lastChangeInForce")
    if last_change:
        meta["Sist endret"] = _text(last_change)

    area = soup.find("dd", class_="legalArea")
    if area:
        # Clean up nested link text: "Fast eiendoms rettsforhold>Husleie" -> nicer
        area_text = _text(area).replace(">", " > ").replace("  ", " ").strip()
        # Collapse "  >" etc.
        area_text = re.sub(r"\s*>\s*", " > ", area_text)
        meta["Rettsområde"] = area_text

    return meta


# ---------------------------------------------------------------------------
# Content conversion
# ---------------------------------------------------------------------------


def _convert_list(ol_tag: Tag) -> str:
    """Convert an <ol>/<ul> with lettered/numbered list items to plain text."""
    lines = []
    for li in ol_tag.find_all("li", recursive=False):
        label = str(li.get("data-name") or "").strip()
        if not label:
            # Fall back to value attribute, or plain bullet for <ul> items
            val = li.get("value", "")
            label = f"{val}." if val else "-"
        # Recurse into the li's content via _convert_paragraph so that
        # nested lists and sub-articles are rendered properly (not run together).
        parts = []
        for child in li.children:
            if not isinstance(child, Tag):
                t = str(child).strip()
                if t:
                    parts.append(t)
                continue
            nested = _convert_paragraph(child)
            if nested:
                parts.append(nested)
        text = " ".join(parts).strip()
        lines.append(f"  {label} {text}")
    return "\n".join(lines)


def _convert_article_content(article: Tag) -> tuple[str, str | None]:
    """Convert an <article class='legalArticle'> to markdown text.

    Returns (content, editorial_note_or_none).
    """
    paragraphs: list[str] = []
    editorial_notes: list[str] = []

    # Iterate over direct children of the article
    for child in article.children:
        if isinstance(child, NavigableString):
            continue
        if not isinstance(child, Tag):
            continue

        # Skip the heading — we handle it separately via _extract_article_title()
        # Lovdata uses h2/h3/h4 depending on nesting level
        if child.name in ("h2", "h3", "h4"):
            continue

        # Editorial / amendment notes
        classes = set(child.get("class") or [])

        if "changesToParent" in classes:
            # Use space separator so link text doesn't run into surrounding text
            # e.g. "Endret ved <a>lov 19 juni...</a>" -> "Endret ved lov 19 juni..."
            note_text = child.get_text(separator=" ", strip=True)
            # Collapse multiple spaces, then remove spurious space before punctuation
            note_text = re.sub(r" +", " ", note_text)
            note_text = re.sub(r" ([),.])", r"\1", note_text)
            if note_text:
                editorial_notes.append(note_text)
            continue

        # Regular content paragraph (may be nested <article class="legalP">
        # or a direct <article> containing sub-articles)
        if child.name == "article":
            para_text = _convert_paragraph(child)
            if para_text:
                paragraphs.append(para_text)
            continue

    content = "\n\n".join(paragraphs)
    editorial = None
    if editorial_notes:
        editorial = "\n\n".join(editorial_notes)

    return content, editorial


def _convert_paragraph(para: Tag) -> str:
    """Convert a single legalP paragraph, handling nested lists."""
    parts: list[str] = []

    for child in para.children:
        if isinstance(child, NavigableString):
            text = str(child).strip()
            if text:
                parts.append(text)
            continue
        if not isinstance(child, Tag):
            continue

        # Nested list (ol.defaultList or ul.defaultList)
        if child.name in ("ol", "ul"):
            parts.append("\n" + _convert_list(child))
            continue

        # Nested legalP inside a listArticle — recurse
        classes = set(child.get("class") or [])

        if "changesToParent" in classes:
            # Skip editorial notes inside paragraphs — handled at article level
            continue

        # Nested sub-article (legalP inside legalP, or listArticle)
        if child.name == "article":
            nested = _convert_paragraph(child)
            if nested:
                parts.append(nested)
            continue

        # Any other tag — just get text (handles <a>, <i>, <b>, <span>, etc.)
        text = _text(child)
        if text:
            parts.append(text)

    return " ".join(parts).strip()


def _extract_article_title(article: Tag) -> str | None:
    """Extract the title from an <article>'s heading element.

    Lovdata uses h3 for most articles, h4 inside nested sub-chapters, and
    h2 in some flat-structure files.  All carry class="legalArticleHeader".
    """
    heading = article.find(["h2", "h3", "h4"], class_="legalArticleHeader")
    if not heading:
        # Fallback: any h2/h3/h4 directly inside this article
        heading = article.find(["h2", "h3", "h4"])
    if not heading:
        return None
    # Reconstruct: "§ X-Y. Title text"
    value_span = heading.find("span", class_="legalArticleValue")
    title_span = heading.find("span", class_="legalArticleTitle")
    if value_span and title_span:
        return f"{_text(value_span)}. {_text(title_span)}"
    if value_span:
        return _text(value_span)
    return _text(heading)


def convert_file(xml_path: Path) -> str:
    """Convert a single Lovdata XML file to markdown-formatted text."""
    content = xml_path.read_text(encoding="utf-8")
    if not content.strip():
        logger.warning("Empty file: %s", xml_path)
        return ""

    soup = BeautifulSoup(content, "html.parser")
    filename = xml_path.stem

    # --- Frontmatter ---
    meta = _extract_metadata(soup, filename)
    lines: list[str] = ["---"]
    for key, value in meta.items():
        lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append("")

    # --- Law title ---
    law_title = meta.get("Tittel", "Ukjent lov")
    lines.append(f"# {law_title}")
    lines.append("")

    # --- Body: chapters and articles ---
    # Walk the document body in DOM order so that root-level articles that appear
    # alongside chapter <section> elements are not silently dropped.
    # We collect the relevant top-level nodes from <main id="dokument"> if present,
    # falling back to the whole soup.
    body_root = soup.find("main", id="dokument") or soup.find("body") or soup
    _convert_body(body_root, lines)

    return "\n".join(lines).rstrip() + "\n"


_TOP_SECTION_RE = re.compile(r"^kapittel-\d+[a-zA-Z]?$")


def _convert_body(root: Tag, lines: list[str]) -> None:
    """Walk the document body in DOM order, emitting chapters and articles.

    This handles three layouts found in Lovdata files:
    1. Pure chapters  — only <section id="kapittel-N"> at root, no stray articles.
    2. Mixed          — <article class="legalArticle"> at root alongside sections
                        (common in older laws where general provisions precede chapters).
    3. Pure flat      — no sections at all, only root-level <article> elements.
    """
    seen_section_ids: set[str] = set()

    for child in root.descendants if False else root.children:  # iterate direct children
        if not isinstance(child, Tag):
            continue

        if child.name == "section":
            raw_id = child.get("id")
            sid = " ".join(raw_id) if isinstance(raw_id, list) else (raw_id or "")
            # Only process top-level chapter sections here; sub-sections are handled
            # recursively inside _convert_section / _convert_subsection.
            if _TOP_SECTION_RE.match(sid):
                seen_section_ids.add(sid)
                _convert_section(child, lines)
            # Non-matching sections (e.g. <section class="footnotes">) are skipped.

        elif child.name == "article":
            classes = set(child.get("class") or [])
            if classes & _LEGAL_ARTICLE_CLASSES:
                _convert_legal_article(child, lines)
            elif "changesToParent" in classes:
                note_text = re.sub(r" +", " ", child.get_text(separator=" ", strip=True))
                note_text = re.sub(r" ([),.])", r"\1", note_text)
                if note_text:
                    lines.append(f"> Endringshistorikk: {note_text}")
                    lines.append("")
            elif "legalP" in classes or "defaultP" in classes or "centeredP" in classes:
                # Root-level prose paragraph (e.g. single-article decrees, farmakopé refs,
                # preamble lines in Grunnloven).
                # Render as a plain paragraph — no ### heading.
                para = _convert_paragraph(child)
                if para:
                    lines.append(para)
                    lines.append("")

        # h1 title, header, footer, nav etc. are all skipped


def _convert_section(section: Tag, lines: list[str]) -> None:
    """Convert a chapter <section> to markdown lines.

    Handles nested sub-sections (e.g. kapittel-3-kapittel-1) by recursing
    into child <section> elements.  Sub-section articles flow directly under
    the parent chapter heading with no extra heading level.
    """
    h2 = section.find("h2", recursive=False)
    if h2:
        lines.append(f"## {_text(h2)}")
        lines.append("")

    for child in section.children:
        if not isinstance(child, Tag):
            continue
        if child.name == "section":
            # Nested sub-section: recurse, emitting named titles as #### if present
            _convert_subsection(child, lines)
        elif child.name == "article":
            classes = set(child.get("class") or [])
            if classes & _LEGAL_ARTICLE_CLASSES:
                _convert_legal_article(child, lines)
            elif "marginIdArticle" in classes:
                line = _convert_margin_id_article(child)
                if line:
                    lines.append(line)
            elif "document-change" in classes:
                _convert_document_change(child, lines)
            elif "changesToParent" in classes:
                # Chapter-level amendment note (e.g. "Overskrift endret ved lov...")
                note_text = re.sub(r" +", " ", child.get_text(separator=" ", strip=True))
                note_text = re.sub(r" ([),.])", r"\1", note_text)
                if note_text:
                    lines.append(f"> Endringshistorikk: {note_text}")
                    lines.append("")
            elif "defaultP" in classes or "legalP" in classes:
                para = _convert_paragraph(child)
                if para:
                    lines.append(para)
                    lines.append("")


def _convert_subsection(section: Tag, lines: list[str]) -> None:
    """Recurse into a nested sub-section, collecting legalArticle children.

    Some sub-sections carry a named title in an unclassed <h3> element
    (e.g. Skatteloven's "Hvem som har skatteplikt og skattepliktens omfang").
    These are rendered as #### headings to preserve document structure.
    Sub-sections without a title (e.g. in Legemiddelforskriften) emit no heading.
    """
    for child in section.children:
        if not isinstance(child, Tag):
            continue
        if child.name in ("h2", "h3", "h4"):
            # Sub-section title (no class = organizational, not an article heading)
            cls = child.get("class") or []
            if isinstance(cls, str):
                cls = [cls]
            if "legalArticleHeader" not in cls:
                title_text = _text(child)
                if title_text:
                    lines.append(f"#### {title_text}")
                    lines.append("")
        elif child.name == "section":
            _convert_subsection(child, lines)
        elif child.name == "article":
            classes = set(child.get("class") or [])
            if classes & _LEGAL_ARTICLE_CLASSES:
                _convert_legal_article(child, lines)
            elif "marginIdArticle" in classes:
                line = _convert_margin_id_article(child)
                if line:
                    lines.append(line)
            elif "changesToParent" in classes:
                # Sub-section-level amendment note (e.g. "Overskrift endret ved lov...")
                note_text = re.sub(r" +", " ", child.get_text(separator=" ", strip=True))
                note_text = re.sub(r" ([),.])", r"\1", note_text)
                if note_text:
                    lines.append(f"> Endringshistorikk: {note_text}")
                    lines.append("")
            elif "defaultP" in classes or "legalP" in classes:
                para = _convert_paragraph(child)
                if para:
                    lines.append(para)
                    lines.append("")


def _convert_document_change(article: Tag, lines: list[str]) -> None:
    """Convert a <article class='document-change'> block to markdown lines.

    These appear in amendment laws and contain:
    - defaultP: introductory prose ("I lov X gjøres følgende endringer:")
    - change: each individual amendment, wrapping a futureLegalArticle
    - More defaultP: separator lines like "– – –"
    """
    for child in article.children:
        if not isinstance(child, Tag):
            continue
        classes = set(child.get("class") or [])
        if "defaultP" in classes:
            para = _convert_paragraph(child)
            if para:
                lines.append(para)
                lines.append("")
        elif "change" in classes:
            # Each change wraps one or more futureLegalArticle elements
            for sub in child.children:
                if not isinstance(sub, Tag):
                    continue
                sub_classes = set(sub.get("class") or [])
                if sub_classes & _LEGAL_ARTICLE_CLASSES:
                    _convert_legal_article(sub, lines)


def _convert_margin_id_article(article: Tag) -> str:
    """Convert a <article class='marginIdArticle'> to a plain-text line.

    Structure: <span class="data-marginOriginalId">1.</span> + nested legalP children.
    Rendered as: "  1. <text>" — same indentation style as list items.
    """
    label_span = article.find("span", class_="data-marginOriginalId")
    label = label_span.get_text(strip=True) if label_span else ""
    # Collect text from nested article children (legalP etc.)
    parts: list[str] = []
    for child in article.children:
        if not isinstance(child, Tag):
            continue
        if child.name == "span":
            continue  # already captured as label
        if child.name == "article":
            text = _convert_paragraph(child)
            if text:
                parts.append(text)
    content = " ".join(parts).strip()
    if label and content:
        return f"  {label} {content}"
    return content or label


def _convert_legal_article(article: Tag, lines: list[str]) -> None:
    """Convert a single legal article to markdown lines."""
    title = _extract_article_title(article)
    if title:
        lines.append(f"### {title}")
        lines.append("")

    content, editorial = _convert_article_content(article)
    if content:
        lines.append(content)
        lines.append("")

    if editorial:
        # Render each editorial line as a blockquote
        for note_line in editorial.split("\n"):
            note_line = note_line.strip()
            if note_line:
                lines.append(f"> Endringshistorikk: {note_line}")
        lines.append("")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Lovdata XML/HTML files to markdown .txt files."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Single XML file to convert (for development).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory of XML files to convert (batch mode).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to data/txt/<nl|sf>/ based on input.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.input_file and not args.input_dir:
        parser.error("Provide --input-file or --input-dir")

    if args.input_file:
        xml_path = args.input_file
        if not xml_path.exists():
            logger.error("File not found: %s", xml_path)
            sys.exit(1)

        output_dir = args.output_dir
        if not output_dir:
            # Default: data/txt/<parent_dir_name>/
            output_dir = xml_path.parent.parent / "txt" / xml_path.parent.name
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = output_dir / f"{xml_path.stem}.txt"
        result = convert_file(xml_path)
        out_path.write_text(result, encoding="utf-8")
        logger.info("Wrote %s (%d bytes)", out_path, len(result.encode("utf-8")))

    elif args.input_dir:
        input_dir = args.input_dir
        if not input_dir.is_dir():
            logger.error("Not a directory: %s", input_dir)
            sys.exit(1)

        output_dir = args.output_dir
        if not output_dir:
            output_dir = input_dir.parent / "txt" / input_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        xml_files = sorted(input_dir.glob("*.xml"))
        if not xml_files:
            logger.warning("No XML files found in %s", input_dir)
            sys.exit(1)

        logger.info("Converting %d files from %s -> %s", len(xml_files), input_dir, output_dir)
        errors = 0
        for i, xml_path in enumerate(xml_files, 1):
            try:
                result = convert_file(xml_path)
                out_path = output_dir / f"{xml_path.stem}.txt"
                out_path.write_text(result, encoding="utf-8")
                if i % 100 == 0 or i == len(xml_files):
                    logger.info("Progress: %d/%d files", i, len(xml_files))
            except Exception as e:
                logger.error("Failed to convert %s: %s", xml_path, e)
                errors += 1

        logger.info("Done. %d files converted, %d errors.", len(xml_files) - errors, errors)


if __name__ == "__main__":
    main()
