import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.application.dto.batch_upload_dto import (
    BatchProcessDocumentRequest,
    BatchProcessDocumentResponse,
    DocumentProcessingResult,
)
from src.application.dto.upload_dto import (
    ProcessDocumentRequest,
    ProcessDocumentResponse,
)
from src.application.use_cases.process_document import ProcessDocumentUseCase
from src.domain.entities.metadata import Metadata

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]  # (current, total, filename)


class BatchProcessDocumentUseCase:
    """Use case for processing multiple documents in batch."""

    def __init__(
        self,
        process_document_use_case: ProcessDocumentUseCase,
        max_workers: int = 4,
    ):
        self._process_document_use_case = process_document_use_case
        self._max_workers = max_workers

    def _process_single(
        self, doc_request: ProcessDocumentRequest
    ) -> DocumentProcessingResult:
        try:
            response = self._process_document_use_case.execute(doc_request)
            return DocumentProcessingResult(
                filename=doc_request.filename,
                response=response,
            )
        except Exception as e:
            logger.exception("Unexpected error processing %s", doc_request.filename)
            return DocumentProcessingResult(
                filename=doc_request.filename,
                response=ProcessDocumentResponse(
                    success=False,
                    metadata=Metadata(pages=0),
                    chunks_created=0,
                    message=f"Unexpected error: {e!s}",
                ),
            )

    def execute(
        self,
        request: BatchProcessDocumentRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> BatchProcessDocumentResponse:
        total = len(request.documents)
        if total == 0:
            return BatchProcessDocumentResponse(
                total_documents=0, successful=0, failed=0, results=[]
            )

        results: list[DocumentProcessingResult | None] = [None] * total
        successful = 0
        failed = 0
        completed_count = 0
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_index = {
                executor.submit(self._process_single, doc): idx
                for idx, doc in enumerate(request.documents)
            }

            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                result = future.result()
                results[idx] = result

                with lock:
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count, total, result.filename)

                if result.response.success:
                    successful += 1
                else:
                    failed += 1

        return BatchProcessDocumentResponse(
            total_documents=total,
            successful=successful,
            failed=failed,
            results=[r for r in results if r is not None],
        )
