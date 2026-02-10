import logging
from collections.abc import Callable

from src.application.dto.batch_upload_dto import (
    BatchProcessDocumentRequest,
    BatchProcessDocumentResponse,
    DocumentProcessingResult,
)
from src.application.dto.upload_dto import ProcessDocumentResponse
from src.application.use_cases.process_document import ProcessDocumentUseCase
from src.domain.entities.metadata import Metadata

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]  # (current, total, filename)


class BatchProcessDocumentUseCase:
    """Use case for processing multiple documents in batch."""

    def __init__(self, process_document_use_case: ProcessDocumentUseCase):
        self._process_document_use_case = process_document_use_case

    def execute(
        self,
        request: BatchProcessDocumentRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> BatchProcessDocumentResponse:
        """
        Process multiple documents, collecting results for each.

        Processing is independent - failures don't affect other documents.

        Args:
            request: Batch request containing list of documents
            progress_callback: Optional callback for progress updates (current, total, filename)

        Returns:
            BatchProcessDocumentResponse with aggregated results
        """
        results: list[DocumentProcessingResult] = []
        successful = 0
        failed = 0
        total = len(request.documents)

        for idx, doc_request in enumerate(request.documents):
            # Notify progress
            if progress_callback:
                progress_callback(idx + 1, total, doc_request.filename)

            try:
                response = self._process_document_use_case.execute(doc_request)

                if response.success:
                    successful += 1
                else:
                    failed += 1

                results.append(
                    DocumentProcessingResult(
                        filename=doc_request.filename,
                        response=response,
                    )
                )

            except Exception as e:
                logger.exception("Unexpected error processing %s", doc_request.filename)
                failed += 1
                results.append(
                    DocumentProcessingResult(
                        filename=doc_request.filename,
                        response=ProcessDocumentResponse(
                            success=False,
                            metadata=Metadata(pages=0),
                            chunks_created=0,
                            message=f"Unexpected error: {e!s}",
                        ),
                    )
                )

        return BatchProcessDocumentResponse(
            total_documents=total,
            successful=successful,
            failed=failed,
            results=results,
        )
