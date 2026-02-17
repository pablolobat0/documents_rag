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

    def collection_exists(self) -> bool:
        """Check whether the underlying collection exists."""
        ...

    def delete_collection(self) -> None:
        """Delete the underlying collection."""
        ...

    def count_chunks(self, filters: dict[str, str | int | list[str]]) -> int:
        """Count chunks matching the given filters."""
        ...
