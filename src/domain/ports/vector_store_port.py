from typing import Any, Protocol


class VectorStorePort(Protocol):
    """Port for vector storage operations."""

    def upsert(
        self,
        chunks: list[str],
        metadata: dict | None = None,
    ) -> None:
        """Insert or update document chunks."""
        ...

    def search(self, query: str) -> list[Any]:
        """Get relevant documents."""
        ...
