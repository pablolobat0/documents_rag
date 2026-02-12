import pytest

from src.infrastructure.processing.markdown_processor import MarkdownProcessor


@pytest.fixture
def processor():
    return MarkdownProcessor()


class TestParseFrontmatter:
    def test_valid_yaml_extracted_and_body_stripped(self, processor):
        text = "---\ntitle: Hello\nauthor: Jane\n---\nBody text"
        frontmatter, body = processor._parse_frontmatter(text)
        assert frontmatter == {"title": "Hello", "author": "Jane"}
        assert body == "Body text"

    def test_no_frontmatter_returns_empty_dict(self, processor):
        text = "No frontmatter here"
        frontmatter, body = processor._parse_frontmatter(text)
        assert frontmatter == {}
        assert body == text

    def test_invalid_yaml_returns_empty_dict_and_full_text(self, processor):
        text = "---\n: invalid: yaml: [:\n---\nBody"
        frontmatter, body = processor._parse_frontmatter(text)
        assert frontmatter == {}
        assert body == text

    def test_non_dict_yaml_ignored(self, processor):
        text = "---\n- item1\n- item2\n---\nBody"
        frontmatter, body = processor._parse_frontmatter(text)
        assert frontmatter == {}
        assert body == text

    def test_list_values_in_frontmatter_preserved(self, processor):
        text = "---\ntags:\n  - python\n  - testing\n---\nBody"
        frontmatter, _body = processor._parse_frontmatter(text)
        assert frontmatter["tags"] == ["python", "testing"]


class TestSplitByHeaders:
    def test_h1_h2_splitting(self, processor):
        text = "# Header 1\nContent 1\n## Header 2\nContent 2"
        sections = processor._split_by_headers(text)
        assert len(sections) == 2
        assert sections[0][0] == "Header 1"
        assert "Content 1" in sections[0][1]
        assert sections[1][0] == "Header 2"
        assert "Content 2" in sections[1][1]

    def test_no_headers_whole_text_as_single_section(self, processor):
        text = "Just plain text without headers."
        sections = processor._split_by_headers(text)
        assert len(sections) == 1
        assert sections[0][0] is None
        assert sections[0][1] == text

    def test_h3_not_treated_as_splitter(self, processor):
        text = "# Main\n### Sub-sub\nContent"
        sections = processor._split_by_headers(text)
        assert len(sections) == 1
        assert "### Sub-sub" in sections[0][1]

    def test_empty_string_returns_single_empty_section(self, processor):
        sections = processor._split_by_headers("")
        assert len(sections) == 1
        assert sections[0][0] is None
        assert sections[0][1] == ""

    def test_content_before_first_header_gets_none_title(self, processor):
        text = "Preamble text\n# Header\nBody"
        sections = processor._split_by_headers(text)
        assert len(sections) == 2
        assert sections[0][0] is None
        assert "Preamble text" in sections[0][1]
        assert sections[1][0] == "Header"


class TestExtractContent:
    def test_full_doc_with_frontmatter(self, processor, markdown_with_frontmatter):
        result = processor.extract_content(markdown_with_frontmatter)
        assert result.total_pages == 2
        assert result.document_metadata["title"] == "Test Document"
        assert result.document_metadata["author"] == "John Doe"
        assert result.page_contents[0].section_title == "Introduction"
        assert result.page_contents[1].section_title == "Details"

    def test_without_frontmatter(self, processor, markdown_without_frontmatter):
        result = processor.extract_content(markdown_without_frontmatter)
        assert result.total_pages == 2
        assert result.document_metadata == {}

    def test_empty_file_returns_zero_pages(self, processor):
        result = processor.extract_content(b"")
        assert result.total_pages == 0
        assert result.page_contents == []

    def test_unicode_content(self, processor):
        content = "# Título\n\nContenido con acentos y ñ".encode()
        result = processor.extract_content(content)
        assert result.total_pages == 1
        assert "ñ" in result.page_contents[0].content

    def test_no_headers(self, processor, markdown_no_headers):
        result = processor.extract_content(markdown_no_headers)
        assert result.total_pages == 1
        assert result.page_contents[0].section_title is None

    def test_empty_sections_excluded(self, processor):
        content = b"# Header 1\n\n# Header 2\n\nActual content here"
        result = processor.extract_content(content)
        # Only sections with non-empty content should be included
        for page in result.page_contents:
            assert page.content.strip() != ""
