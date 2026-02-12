from dataclasses import dataclass, field

from src.application.dto.upload_dto import (
    ProcessDocumentRequest,
    ProcessDocumentResponse,
)


@dataclass
class BatchProcessDocumentRequest:
    """Request for processing multiple documents."""

    documents: list[ProcessDocumentRequest]


@dataclass
class DocumentProcessingResult:
    """Result for a single document in a batch."""

    filename: str
    response: ProcessDocumentResponse


@dataclass
class BatchProcessDocumentResponse:
    """Response for batch document processing."""

    total_documents: int
    successful: int
    failed: int
    results: list[DocumentProcessingResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.successful / self.total_documents) * 100

    @property
    def all_succeeded(self) -> bool:
        """Check if all documents were processed successfully."""
        return self.failed == 0 and self.total_documents > 0

    @property
    def total_chunks_created(self) -> int:
        """Calculate total chunks created across all successful documents."""
        return sum(
            r.response.chunks_created for r in self.results if r.response.success
        )
