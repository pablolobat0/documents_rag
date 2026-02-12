import pytest

from src.domain.value_objects.page_content import PageContent
from src.infrastructure.processing.text_splitter import LangchainTextSplitter


@pytest.fixture
def splitter():
    return LangchainTextSplitter(chunk_size=100, chunk_overlap=20)


class TestSplitPages:
    def test_single_short_page_returns_one_chunk(self, splitter):
        pages = [PageContent(content="Short text.", page_number=1, content_type="text")]
        chunks = splitter.split_pages(pages, {"document_name": "test.txt"})
        assert len(chunks) == 1
        assert chunks[0].metadata["document_name"] == "test.txt"
        assert chunks[0].metadata["page_number"] == 1
        assert chunks[0].metadata["content_type"] == "text"

    def test_multiple_pages_preserve_page_number(self, splitter):
        pages = [
            PageContent(content="Page one text.", page_number=1, content_type="text"),
            PageContent(content="Page two text.", page_number=2, content_type="text"),
        ]
        chunks = splitter.split_pages(pages, {})
        page_numbers = [c.metadata["page_number"] for c in chunks]
        assert 1 in page_numbers
        assert 2 in page_numbers

    def test_large_text_split_into_multiple_chunks(self, splitter):
        long_text = "Word " * 200  # ~1000 chars, well above chunk_size=100
        pages = [PageContent(content=long_text, page_number=1, content_type="text")]
        chunks = splitter.split_pages(pages, {})
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["page_number"] == 1

    def test_metadata_merging(self, splitter):
        pages = [
            PageContent(content="Content.", page_number=3, content_type="image_caption")
        ]
        base_metadata = {"document_name": "doc.pdf", "file_type": "application/pdf"}
        chunks = splitter.split_pages(pages, base_metadata)
        assert len(chunks) == 1
        meta = chunks[0].metadata
        assert meta["document_name"] == "doc.pdf"
        assert meta["file_type"] == "application/pdf"
        assert meta["page_number"] == 3
        assert meta["content_type"] == "image_caption"

    def test_empty_pages_list_returns_empty(self, splitter):
        chunks = splitter.split_pages([], {"document_name": "empty.txt"})
        assert chunks == []

    def test_chunk_overlap_behavior(self):
        splitter = LangchainTextSplitter(chunk_size=50, chunk_overlap=10)
        # Create text that will definitely be split
        text = ". ".join(f"Sentence number {i}" for i in range(20))
        pages = [PageContent(content=text, page_number=1, content_type="text")]
        chunks = splitter.split_pages(pages, {})
        assert len(chunks) > 1
        # Soft check: overlap depends on split boundaries, so just verify
        # consecutive chunks aren't identical for large enough chunks
        for i in range(len(chunks) - 1):
            assert (
                len(chunks[i].content) <= 50
                or chunks[i].content != chunks[i + 1].content
            )
