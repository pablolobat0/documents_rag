from typing import Protocol

from src.domain.value_objects.retrieved_document import RetrievedDocument


class VectorStorePort(Protocol):
    """Port for vector storage operations."""

    def upsert(
        self,
        chunks: list[str],
        metadata: dict | None = None,
    ) -> None:
        """Insert or update document chunks."""
        ...

    def search(self, query: str) -> list[RetrievedDocument]:
        """Get relevant documents."""
        ...
