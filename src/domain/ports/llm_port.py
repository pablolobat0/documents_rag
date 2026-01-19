from typing import Any, Protocol


class LLMPort(Protocol):
    """Port for LLM operations."""

    def invoke(self, input: Any) -> Any:
        """Invoke LLM with input and return response."""
        ...

    def with_structured_output(self, schema: type) -> Any:
        """Return a runnable that outputs structured data matching the schema."""
        ...


class EmbeddingsPort(Protocol):
    """Port for embeddings operations."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for documents."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query."""
        ...
