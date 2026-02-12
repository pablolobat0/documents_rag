from unittest.mock import MagicMock

import pytest

from src.application.dto.upload_dto import ProcessDocumentRequest
from src.application.use_cases.process_document import ProcessDocumentUseCase
from src.domain.entities.metadata import Metadata
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
            frontmatter={"tags": ["python", "testing", "docs"]},
        )
        result = use_case._build_chunk_metadata(metadata)
        assert result["tags"] == "python, testing, docs"

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
