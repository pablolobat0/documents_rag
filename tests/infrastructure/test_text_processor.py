from src.infrastructure.processing.text_processor import PlainTextProcessor


class TestPlainTextProcessor:
    def test_returns_single_page_extraction_result(self):
        processor = PlainTextProcessor()
        result = processor.extract_content(b"Hello, world!")
        assert result.total_pages == 1
        assert len(result.page_contents) == 1
        assert result.page_contents[0].content == "Hello, world!"
        assert result.page_contents[0].page_number == 1
        assert result.page_contents[0].content_type == "text"

    def test_supported_content_types(self):
        processor = PlainTextProcessor()
        assert processor.supported_content_types == ["text/plain"]
