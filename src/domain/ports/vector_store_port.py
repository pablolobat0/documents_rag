from typing import Protocol

from src.domain.value_objects.document_chunk import DocumentChunk
from src.domain.value_objects.retrieved_document import RetrievedDocument


class VectorStorePort(Protocol):
    """Port for vector storage operations."""

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Insert or update document chunks with per-chunk metadata."""
        ...

    def search(
        self,
        query: str,
        num_documents: int,
        filters: dict[str, str | list[str]] | None = None,
    ) -> list[RetrievedDocument]:
        """Get relevant documents."""
        ...
