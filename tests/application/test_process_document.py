from unittest.mock import MagicMock

import pytest

from src.application.dto.upload_dto import ProcessDocumentRequest
from src.application.use_cases.process_document import ProcessDocumentUseCase
from src.domain.entities.metadata import Metadata
from src.domain.value_objects.document_classification import DocumentClassification
from src.domain.value_objects.extraction_result import ExtractionResult
from src.domain.value_objects.page_content import PageContent


@pytest.fixture
def use_case(mock_vector_store, mock_text_splitter, mock_registry_with_extractor):
    return ProcessDocumentUseCase(
        vector_store=mock_vector_store,
        content_extractor_registry=mock_registry_with_extractor,
        text_splitter=mock_text_splitter,
    )


@pytest.fixture
def valid_request():
    return ProcessDocumentRequest(
        content=b"hello world",
        filename="test.txt",
        content_type="text/plain",
    )


class TestExecute:
    def test_success_path_calls_upsert(
        self, use_case, valid_request, mock_vector_store
    ):
        response = use_case.execute(valid_request)
        assert response.success is True
        assert response.chunks_created == 2
        mock_vector_store.upsert_chunks.assert_called_once()

    def test_unsupported_content_type(self, use_case):
        request = ProcessDocumentRequest(
            content=b"data",
            filename="file.xyz",
            content_type="application/xyz",
        )
        response = use_case.execute(request)
        assert response.success is False
        assert "Unsupported content type" in response.message

    def test_empty_extracted_content(
        self, mock_vector_store, mock_text_splitter, valid_request
    ):
        registry = MagicMock()
        extractor = MagicMock()
        extractor.extract_content.return_value = ExtractionResult(
            page_contents=[], total_pages=0
        )
        registry.get_extractor.return_value = extractor

        uc = ProcessDocumentUseCase(mock_vector_store, registry, mock_text_splitter)
        response = uc.execute(valid_request)
        assert response.success is False
        assert "No content found" in response.message

    def test_no_chunks_created(
        self, mock_vector_store, mock_registry_with_extractor, valid_request
    ):
        splitter = MagicMock()
        splitter.split_pages.return_value = []

        uc = ProcessDocumentUseCase(
            mock_vector_store, mock_registry_with_extractor, splitter
        )
        response = uc.execute(valid_request)
        assert response.success is False
        assert "No chunks created" in response.message

    def test_exception_returns_error_response(
        self, mock_vector_store, mock_text_splitter
    ):
        registry = MagicMock()
        registry.get_extractor.side_effect = RuntimeError("boom")

        uc = ProcessDocumentUseCase(mock_vector_store, registry, mock_text_splitter)
        request = ProcessDocumentRequest(
            content=b"data", filename="test.txt", content_type="text/plain"
        )
        response = uc.execute(request)
        assert response.success is False
        assert "boom" in response.message


class TestBuildChunkMetadata:
    def test_flat_base_metadata(self, use_case):
        metadata = Metadata(
            pages=5,
            document_name="doc.pdf",
            file_type="application/pdf",
            file_size=1024,
        )
        result = use_case._build_chunk_metadata(metadata)
        assert result["document_name"] == "doc.pdf"
        assert result["file_type"] == "application/pdf"
        assert result["file_size"] == 1024
        assert result["total_pages"] == 5

    def test_frontmatter_lists_comma_separated(self, use_case):
        metadata = Metadata(
            pages=1,
            frontmatter={"categories": ["python", "testing", "docs"]},
        )
        result = use_case._build_chunk_metadata(metadata)
        assert result["categories"] == "python, testing, docs"

    def test_frontmatter_type_and_tags_skipped(self, use_case):
        metadata = Metadata(
            pages=1,
            frontmatter={"type": "book", "tags": ["AI"], "author": "Jane"},
        )
        result = use_case._build_chunk_metadata(metadata)
        assert "type" not in result
        assert "tags" not in result
        assert result["author"] == "Jane"

    def test_classification_added_to_metadata(self, use_case):
        metadata = Metadata(pages=1)
        classification = DocumentClassification(type="book", tags=["AI", "LLM"])
        result = use_case._build_chunk_metadata(metadata, classification)
        assert result["type"] == "book"
        assert result["tags"] == ["AI", "LLM"]

    def test_tags_stored_as_list(self, use_case):
        metadata = Metadata(pages=1)
        classification = DocumentClassification(tags=["rag", "transformers"])
        result = use_case._build_chunk_metadata(metadata, classification)
        assert isinstance(result["tags"], list)
        assert result["tags"] == ["rag", "transformers"]

    def test_scalar_frontmatter_preserved(self, use_case):
        metadata = Metadata(
            pages=1,
            frontmatter={"author": "Jane", "version": 2},
        )
        result = use_case._build_chunk_metadata(metadata)
        assert result["author"] == "Jane"
        assert result["version"] == 2

    def test_empty_frontmatter(self, use_case):
        metadata = Metadata(pages=1)
        result = use_case._build_chunk_metadata(metadata)
        # No frontmatter keys should be added
        assert "tags" not in result
        assert "author" not in result


class TestExtractClassification:
    @pytest.fixture
    def use_case_with_classifier(
        self, mock_vector_store, mock_text_splitter, mock_registry_with_extractor
    ):
        classifier = MagicMock()
        return ProcessDocumentUseCase(
            vector_store=mock_vector_store,
            content_extractor_registry=mock_registry_with_extractor,
            text_splitter=mock_text_splitter,
            metadata_classifier=classifier,
        )

    def test_markdown_valid_frontmatter(self, use_case):
        pages = [PageContent(content="text", page_number=1, content_type="text")]
        result = use_case._extract_classification(
            "text/markdown", pages, {"type": "book", "tags": ["AI"]}
        )
        assert result.type == "book"
        assert result.tags == ["AI"]

    def test_markdown_invalid_frontmatter_returns_empty(self, use_case):
        pages = [PageContent(content="text", page_number=1, content_type="text")]
        result = use_case._extract_classification(
            "text/markdown", pages, {"type": "invalid_type"}
        )
        assert result.type is None
        assert result.tags is None

    def test_non_markdown_calls_classifier(self, use_case_with_classifier):
        pages = [
            PageContent(content="content about AI", page_number=1, content_type="text")
        ]
        expected = DocumentClassification(type="concept", tags=["AI"])
        use_case_with_classifier._metadata_classifier.classify.return_value = expected

        result = use_case_with_classifier._extract_classification(
            "application/pdf", pages, {}
        )
        assert result == expected
        use_case_with_classifier._metadata_classifier.classify.assert_called_once()

    def test_no_classifier_returns_empty(self, use_case):
        pages = [PageContent(content="content", page_number=1, content_type="text")]
        result = use_case._extract_classification("application/pdf", pages, {})
        assert result.type is None
        assert result.tags is None

    def test_markdown_without_type_tags_uses_classifier(self, use_case_with_classifier):
        pages = [PageContent(content="text", page_number=1, content_type="text")]
        expected = DocumentClassification(type="prompt")
        use_case_with_classifier._metadata_classifier.classify.return_value = expected

        result = use_case_with_classifier._extract_classification(
            "text/markdown", pages, {"author": "Jane"}
        )
        # No type/tags in frontmatter, so falls through to classifier
        assert result == expected

    def test_uses_first_3_pages_for_classifier(self, use_case_with_classifier):
        pages = [
            PageContent(content=f"page {i}", page_number=i, content_type="text")
            for i in range(5)
        ]
        use_case_with_classifier._metadata_classifier.classify.return_value = (
            DocumentClassification()
        )

        use_case_with_classifier._extract_classification("application/pdf", pages, {})

        call_content = use_case_with_classifier._metadata_classifier.classify.call_args[
            0
        ][0]
        assert "page 0" in call_content
        assert "page 2" in call_content
        assert "page 3" not in call_content
