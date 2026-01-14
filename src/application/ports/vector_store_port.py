from typing import Protocol


class VectorStorePort(Protocol):
    """Port for vector storage operations."""

    def upsert(
        self, chunks: list[str], embeddings: list[list[float]], metadata: dict | None = None
    ) -> None:
        """Insert or update document chunks."""
        ...

    def get_retriever(self, search_type: str = "mmr", k: int = 10):
        """Get a retriever for searching documents."""
        ...
