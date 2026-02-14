"""Tests for the Lovdata HTML parser with hierarchical extraction."""

import pytest
from pathlib import Path
from lovli.parser import (
    parse_xml_file,
    parse_law_header,
    _extract_law_ref_from_filename,
    _extract_short_name,
    _extract_cross_references,
    LegalArticle,
)
from bs4 import BeautifulSoup


# Path to test data (Husleieloven)
HUSLEIELOVEN_PATH = Path(__file__).parent.parent / "data" / "nl" / "nl-19990326-017.xml"
FORBRUKERKJOP_PATH = Path(__file__).parent.parent / "data" / "nl" / "nl-20020621-034.xml"


# --- Unit tests for helper functions ---


class TestExtractLawRef:
    def test_standard_format(self):
        assert _extract_law_ref_from_filename("nl-19990326-017") == "lov/1999-03-26-017"

    def test_different_date(self):
        assert _extract_law_ref_from_filename("nl-20020621-034") == "lov/2002-06-21-034"

    def test_fallback(self):
        assert _extract_law_ref_from_filename("unknown-format") == "unknown-format"


class TestExtractShortName:
    def test_with_separator(self):
        html = '<dd class="titleShort">Husleieloven – husll</dd>'
        soup = BeautifulSoup(html, "html.parser")
        assert _extract_short_name(soup) == "Husleieloven"

    def test_with_dash_separator(self):
        html = '<dd class="titleShort">Forbrukerkjøpsloven - fkjl</dd>'
        soup = BeautifulSoup(html, "html.parser")
        assert _extract_short_name(soup) == "Forbrukerkjøpsloven"

    def test_no_separator(self):
        html = '<dd class="titleShort">Grunnloven</dd>'
        soup = BeautifulSoup(html, "html.parser")
        assert _extract_short_name(soup) == "Grunnloven"

    def test_missing_element(self):
        html = "<div>No short name here</div>"
        soup = BeautifulSoup(html, "html.parser")
        assert _extract_short_name(soup) is None


class TestExtractCrossReferences:
    def test_external_law_refs(self):
        html = """
        <article id="test">
            <p>See <a href="lov/2002-06-21-34">forbrukerkjøpsloven</a> and
            <a href="lov/2005-06-17-90/§5">arbeidsmiljøloven § 5</a></p>
        </article>
        """
        soup = BeautifulSoup(html, "html.parser")
        article = soup.find("article")
        refs = _extract_cross_references(article, "lov/1999-03-26-17")
        assert "lov/2002-06-21-34" in refs
        assert "lov/2005-06-17-90/§5" in refs

    def test_self_refs_excluded(self):
        html = """
        <article id="test">
            <p>See <a href="lov/1999-03-26-17/§3-5">§ 3-5</a></p>
        </article>
        """
        soup = BeautifulSoup(html, "html.parser")
        article = soup.find("article")
        refs = _extract_cross_references(article, "lov/1999-03-26-17")
        assert len(refs) == 0

    def test_self_refs_excluded_with_zero_padding_mismatch(self):
        html = """
        <article id="test">
            <p>
              <a href="lov/1999-03-26-17/§3-5">§ 3-5</a>
              <a href="lov/1999-03-26-17#kapittel-3-paragraf-5">§ 3-5 anchor</a>
            </p>
        </article>
        """
        soup = BeautifulSoup(html, "html.parser")
        article = soup.find("article")
        refs = _extract_cross_references(article, "lov/1999-03-26-017")
        assert refs == []

    def test_forskrift_refs(self):
        html = """
        <article id="test">
            <p>See <a href="forskrift/2009-06-12-641">forskrift</a></p>
        </article>
        """
        soup = BeautifulSoup(html, "html.parser")
        article = soup.find("article")
        refs = _extract_cross_references(article, "lov/1999-03-26-17")
        assert "forskrift/2009-06-12-641" in refs

    def test_deduplication(self):
        html = """
        <article id="test">
            <p><a href="lov/2002-06-21-34">ref1</a> and <a href="lov/2002-06-21-34">ref2</a></p>
        </article>
        """
        soup = BeautifulSoup(html, "html.parser")
        article = soup.find("article")
        refs = _extract_cross_references(article, "lov/1999-03-26-17")
        assert len(refs) == 1

    def test_no_refs(self):
        html = '<article id="test"><p>No links here</p></article>'
        soup = BeautifulSoup(html, "html.parser")
        article = soup.find("article")
        refs = _extract_cross_references(article, "lov/1999-03-26-17")
        assert refs == []

    def test_non_law_links_ignored(self):
        html = """
        <article id="test">
            <p><a href="https://example.com">external</a>
            <a href="legal-areas/08">area</a></p>
        </article>
        """
        soup = BeautifulSoup(html, "html.parser")
        article = soup.find("article")
        refs = _extract_cross_references(article, "lov/1999-03-26-17")
        assert refs == []


def test_fallback_source_anchor_id_is_unique_across_chapters(tmp_path):
    """Fallback source anchors should include chapter scope to avoid collisions."""
    sample = tmp_path / "nl-20990101-001.xml"
    sample.write_text(
        """
        <html>
          <body>
            <dd class="title">Testlov</dd>
            <section id="kapittel-1">
              <h2>Kapittel 1. Innledning</h2>
              <article>
                <h3>Untitled Article</h3>
                <p>Endret ved lov ...</p>
              </article>
            </section>
            <section id="kapittel-2">
              <h2>Kapittel 2. Innledning</h2>
              <article>
                <h3>Untitled Article</h3>
                <p>Endret ved lov ...</p>
              </article>
            </section>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    articles = list(parse_xml_file(sample))
    assert len(articles) == 2
    source_ids = [a.source_anchor_id for a in articles]
    assert len(set(source_ids)) == 2
    assert source_ids[0] == "nl-20990101-001_kapittel-1_art_0"
    assert source_ids[1] == "nl-20990101-001_kapittel-2_art_0"


def test_mixed_structure_keeps_root_level_articles(tmp_path):
    """Root-level substantive articles should be kept even when sections exist."""
    sample = tmp_path / "sf-20990101-001.xml"
    sample.write_text(
        """
        <html>
          <body>
            <dd class="title">Testforskrift</dd>
            <section id="kapittel-1">
              <h2>Kapittel 1. Innledning</h2>
              <article id="kapittel-1-paragraf-1">
                <h3>§ 1. Definisjon</h3>
                <p>Definisjonstekst.</p>
              </article>
            </section>
            <article id="paragraf-2">
              <p>Dette er en selvstendig bestemmelse utenfor section.</p>
            </article>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    articles = list(parse_xml_file(sample))
    ids = {a.article_id for a in articles}
    assert "kapittel-1-paragraf-1" in ids
    assert "paragraf-2" in ids


def test_relaxed_nested_fallback_when_coarse_yields_zero(tmp_path):
    """Parser should keep nested IDs when strict filtering would drop everything."""
    sample = tmp_path / "sf-20990101-002.xml"
    sample.write_text(
        """
        <html>
          <body>
            <dd class="title">Kun nested innhold</dd>
            <article id="kapittel-1-ledd-1">
              <p>Første ledd med materiell regel.</p>
            </article>
            <article id="kapittel-1-ledd-2">
              <p>Andre ledd med materiell regel.</p>
            </article>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    articles = list(parse_xml_file(sample))
    assert len(articles) == 2
    assert {a.source_anchor_id for a in articles} == {"kapittel-1-ledd-1", "kapittel-1-ledd-2"}


def test_fallback_art_doc_type_uses_content_signals(tmp_path):
    """_art_ fallback chunks should not always be classified as editorial."""
    sample = tmp_path / "nl-20990101-003.xml"
    sample.write_text(
        """
        <html>
          <body>
            <dd class="title">Fallback doc type test</dd>
            <section id="kapittel-1">
              <h2>Kapittel 1. Endringer</h2>
              <article>
                <p>§ 15-13 a tredje ledd skal lyde: Dette er operativ lovtekst.</p>
              </article>
              <article>
                <p>Endret ved lov 1 jan 2024 nr. 1.</p>
              </article>
            </section>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    articles = list(parse_xml_file(sample))
    assert len(articles) == 2
    by_source = {a.source_anchor_id: a for a in articles}
    assert by_source["nl-20990101-003_kapittel-1_art_0"].doc_type == "provision"
    assert by_source["nl-20990101-003_kapittel-1_art_1"].doc_type == "editorial_note"


def test_header_article_count_matches_parser_output(tmp_path):
    """Header article_count should use the same extraction logic as parse_xml_file."""
    sample = tmp_path / "sf-20990101-004.xml"
    sample.write_text(
        """
        <html>
          <head><title>Header parity test</title></head>
          <body>
            <header>
              <dl>
                <dd class="title">Header parity test</dd>
                <dd class="dokid">SF/forskrift/2099-01-01-4</dd>
              </dl>
            </header>
            <article id="kapittel-1-ledd-1"><p>Ledd 1.</p></article>
            <article id="kapittel-1-ledd-2"><p>Ledd 2.</p></article>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    parsed = list(parse_xml_file(sample))
    header = parse_law_header(sample)
    assert header["article_count"] == len(parsed)


# --- Integration tests with real law files ---


@pytest.mark.skipif(not HUSLEIELOVEN_PATH.exists(), reason="Husleieloven data file not available")
class TestParseHusleieloven:
    """Integration tests using the real Husleieloven file."""

    def test_articles_extracted(self):
        articles = list(parse_xml_file(HUSLEIELOVEN_PATH))
        assert len(articles) > 50  # Husleieloven has ~93 paragraphs

    def test_law_title(self):
        articles = list(parse_xml_file(HUSLEIELOVEN_PATH))
        assert articles[0].law_title == "Lov om husleieavtaler (husleieloven)"

    def test_short_name_extracted(self):
        articles = list(parse_xml_file(HUSLEIELOVEN_PATH))
        # All articles should have the same short name
        for art in articles:
            assert art.law_short_name is not None
            assert "Husleieloven" in art.law_short_name

    def test_chapter_metadata(self):
        articles = list(parse_xml_file(HUSLEIELOVEN_PATH))
        # Every article should have chapter metadata
        for art in articles:
            assert art.chapter_id is not None, f"Missing chapter_id for {art.article_id}"
            assert art.chapter_title is not None, f"Missing chapter_title for {art.article_id}"

    def test_chapter_id_format(self):
        articles = list(parse_xml_file(HUSLEIELOVEN_PATH))
        for art in articles:
            assert art.chapter_id.startswith("kapittel-"), f"Bad chapter_id: {art.chapter_id}"

    def test_chapter_title_cleaned(self):
        """Chapter titles should not have the 'Kapittel X.' prefix."""
        articles = list(parse_xml_file(HUSLEIELOVEN_PATH))
        for art in articles:
            assert not art.chapter_title.startswith("Kapittel "), (
                f"Chapter title not cleaned: {art.chapter_title}"
            )

    def test_depositum_article(self):
        """Test specific article: § 3-5 Depositum."""
        articles = list(parse_xml_file(HUSLEIELOVEN_PATH))
        depositum = [a for a in articles if a.article_id == "kapittel-3-paragraf-5"]
        assert len(depositum) == 1
        art = depositum[0]
        assert "Depositum" in art.title
        assert art.chapter_id == "kapittel-3"
        assert "seks" in art.content.lower()  # "seks måneders leie"
        assert art.url.endswith("#kapittel-3-paragraf-5")

    def test_cross_references_present(self):
        """Some articles should have cross-references."""
        articles = list(parse_xml_file(HUSLEIELOVEN_PATH))
        articles_with_refs = [a for a in articles if a.cross_references]
        assert len(articles_with_refs) > 0, "No articles have cross-references"

    def test_no_sub_articles(self):
        """Sub-articles (-ledd-, -punkt-) should be filtered out."""
        articles = list(parse_xml_file(HUSLEIELOVEN_PATH))
        for art in articles:
            assert "-ledd-" not in art.article_id
            assert "-punkt-" not in art.article_id

    def test_doc_type_defaults_and_editorial_notes(self):
        """Parser should classify substantive sections vs editorial notes."""
        articles = list(parse_xml_file(HUSLEIELOVEN_PATH))
        assert any(a.doc_type == "provision" for a in articles)
        editorial = [a for a in articles if a.doc_type == "editorial_note"]
        assert editorial, "Expected at least one editorial note chunk in husleieloven"
        assert all(a.law_id == "nl-19990326-017" for a in editorial)

    def test_url_format(self):
        articles = list(parse_xml_file(HUSLEIELOVEN_PATH))
        for art in articles:
            assert art.url.startswith("https://lovdata.no/lov/")
            assert "/lov/1999-03-26-17#" in art.url

    def test_canonical_article_ids_from_heading(self):
        """Article IDs should follow visible § numbering (canonical IDs)."""
        articles = list(parse_xml_file(HUSLEIELOVEN_PATH))
        by_id = {a.article_id: a for a in articles}
        assert "kapittel-9-paragraf-6" in by_id
        assert "kapittel-9-paragraf-7" in by_id
        assert "kapittel-9-paragraf-3a" in by_id

    def test_source_anchor_id_preserved_for_urls(self):
        """Source anchor IDs are preserved even when canonical IDs differ."""
        articles = list(parse_xml_file(HUSLEIELOVEN_PATH))
        oppsigelsesfrist = [a for a in articles if a.article_id == "kapittel-9-paragraf-6"]
        assert len(oppsigelsesfrist) == 1
        art = oppsigelsesfrist[0]
        assert art.source_anchor_id == "kapittel-9-paragraf-7"
        assert art.url.endswith("#kapittel-9-paragraf-7")

    def test_law_id(self):
        articles = list(parse_xml_file(HUSLEIELOVEN_PATH))
        assert all(a.law_id == "nl-19990326-017" for a in articles)


@pytest.mark.skipif(not HUSLEIELOVEN_PATH.exists(), reason="Husleieloven data file not available")
class TestParseLawHeader:
    """Test header-only parsing for catalog building."""

    def test_header_metadata(self):
        header = parse_law_header(HUSLEIELOVEN_PATH)
        assert header["law_id"] == "nl-19990326-017"
        assert "husleie" in header["law_title"].lower()
        assert header["law_short_name"] is not None
        assert header["chapter_count"] == 13
        assert header["article_count"] > 50
        assert header["legal_area"] is not None

    def test_header_law_ref(self):
        header = parse_law_header(HUSLEIELOVEN_PATH)
        assert header["law_ref"] == "lov/1999-03-26-17"


@pytest.mark.skipif(not FORBRUKERKJOP_PATH.exists(), reason="Forbrukerkjopsloven data file not available")
class TestParseForbrukerkjopsloven:
    """Integration tests with a second law to verify consistency."""

    def test_articles_extracted(self):
        articles = list(parse_xml_file(FORBRUKERKJOP_PATH))
        assert len(articles) > 30

    def test_short_name(self):
        articles = list(parse_xml_file(FORBRUKERKJOP_PATH))
        assert articles[0].law_short_name is not None
        assert "Forbrukerkjøpsloven" in articles[0].law_short_name

    def test_chapter_metadata(self):
        articles = list(parse_xml_file(FORBRUKERKJOP_PATH))
        for art in articles:
            assert art.chapter_id is not None
            assert art.chapter_title is not None
