from unittest.mock import MagicMock

import pytest

from src.application.dto.batch_upload_dto import BatchProcessDocumentRequest
from src.application.dto.upload_dto import (
    ProcessDocumentRequest,
    ProcessDocumentResponse,
)
from src.application.use_cases.batch_process_documents import (
    BatchProcessDocumentUseCase,
)
from src.domain.entities.metadata import Metadata


def _make_request(filename: str) -> ProcessDocumentRequest:
    return ProcessDocumentRequest(
        content=b"data", filename=filename, content_type="text/plain"
    )


def _success_response(chunks: int = 5) -> ProcessDocumentResponse:
    return ProcessDocumentResponse(
        success=True, metadata=Metadata(pages=1), chunks_created=chunks
    )


def _failure_response(msg: str = "error") -> ProcessDocumentResponse:
    return ProcessDocumentResponse(
        success=False, metadata=Metadata(pages=0), chunks_created=0, message=msg
    )


@pytest.fixture
def mock_inner_uc():
    return MagicMock()


@pytest.fixture
def use_case(mock_inner_uc):
    return BatchProcessDocumentUseCase(
        process_document_use_case=mock_inner_uc, max_workers=1
    )


class TestBatchExecute:
    def test_all_succeed(self, use_case, mock_inner_uc):
        mock_inner_uc.execute.return_value = _success_response()
        request = BatchProcessDocumentRequest(
            documents=[_make_request("a.txt"), _make_request("b.txt")]
        )
        response = use_case.execute(request)
        assert response.total_documents == 2
        assert response.successful == 2
        assert response.failed == 0

    def test_all_fail(self, use_case, mock_inner_uc):
        mock_inner_uc.execute.return_value = _failure_response()
        request = BatchProcessDocumentRequest(
            documents=[_make_request("a.txt"), _make_request("b.txt")]
        )
        response = use_case.execute(request)
        assert response.successful == 0
        assert response.failed == 2

    def test_mixed_results(self, use_case, mock_inner_uc):
        mock_inner_uc.execute.side_effect = [
            _success_response(),
            _failure_response(),
            _success_response(),
        ]
        request = BatchProcessDocumentRequest(
            documents=[
                _make_request("a.txt"),
                _make_request("b.txt"),
                _make_request("c.txt"),
            ]
        )
        response = use_case.execute(request)
        assert response.successful == 2
        assert response.failed == 1

    def test_exception_does_not_block_others(self, use_case, mock_inner_uc):
        mock_inner_uc.execute.side_effect = [
            RuntimeError("unexpected"),
            _success_response(),
        ]
        request = BatchProcessDocumentRequest(
            documents=[_make_request("a.txt"), _make_request("b.txt")]
        )
        response = use_case.execute(request)
        assert response.successful == 1
        assert response.failed == 1
        assert "unexpected" in response.results[0].response.message

    def test_progress_callback_called(self, use_case, mock_inner_uc):
        mock_inner_uc.execute.return_value = _success_response()
        callback = MagicMock()
        request = BatchProcessDocumentRequest(
            documents=[_make_request("a.txt"), _make_request("b.txt")]
        )
        use_case.execute(request, progress_callback=callback)
        assert len(callback.call_args_list) == 2
        counters = {c.args[0] for c in callback.call_args_list}
        assert counters == {1, 2}
        filenames = {c.args[2] for c in callback.call_args_list}
        assert filenames == {"a.txt", "b.txt"}

    def test_empty_batch(self, use_case, mock_inner_uc):
        request = BatchProcessDocumentRequest(documents=[])
        response = use_case.execute(request)
        assert response.total_documents == 0
        assert response.successful == 0
        assert response.failed == 0
        mock_inner_uc.execute.assert_not_called()
