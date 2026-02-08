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

    def test_url_format(self):
        articles = list(parse_xml_file(HUSLEIELOVEN_PATH))
        for art in articles:
            assert art.url.startswith("https://lovdata.no/lov/")

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
        assert header["law_ref"] == "lov/1999-03-26-017"


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
