import pytest

from src.application.dto.batch_upload_dto import (
    BatchProcessDocumentResponse,
    DocumentProcessingResult,
)
from src.application.dto.upload_dto import ProcessDocumentResponse
from src.domain.entities.metadata import Metadata


def _result(success: bool, chunks: int = 0) -> DocumentProcessingResult:
    return DocumentProcessingResult(
        filename="test.txt",
        response=ProcessDocumentResponse(
            success=success,
            metadata=Metadata(pages=1),
            chunks_created=chunks,
        ),
    )


class TestSuccessRate:
    def test_100_percent(self):
        resp = BatchProcessDocumentResponse(total_documents=4, successful=4, failed=0)
        assert resp.success_rate == 100.0

    def test_75_percent(self):
        resp = BatchProcessDocumentResponse(total_documents=4, successful=3, failed=1)
        assert resp.success_rate == 75.0

    def test_zero_total_returns_zero(self):
        resp = BatchProcessDocumentResponse(total_documents=0, successful=0, failed=0)
        assert resp.success_rate == 0.0


class TestAllSucceeded:
    def test_true_when_all_succeed(self):
        resp = BatchProcessDocumentResponse(total_documents=3, successful=3, failed=0)
        assert resp.all_succeeded is True

    def test_false_with_failures(self):
        resp = BatchProcessDocumentResponse(total_documents=3, successful=2, failed=1)
        assert resp.all_succeeded is False

    def test_false_when_empty(self):
        resp = BatchProcessDocumentResponse(total_documents=0, successful=0, failed=0)
        assert resp.all_succeeded is False


class TestTotalChunksCreated:
    def test_sum_across_successes(self):
        resp = BatchProcessDocumentResponse(
            total_documents=3,
            successful=2,
            failed=1,
            results=[_result(True, 5), _result(True, 3), _result(False, 0)],
        )
        assert resp.total_chunks_created == 8

    def test_zero_when_empty(self):
        resp = BatchProcessDocumentResponse(
            total_documents=0, successful=0, failed=0, results=[]
        )
        assert resp.total_chunks_created == 0
