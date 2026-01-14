from typing import Any, Protocol


class LLMPort(Protocol):
    """Port for LLM operations."""

    def invoke(self, messages: list[Any]) -> Any:
        """Invoke LLM with messages."""
        ...

    def with_structured_output(self, schema: type) -> "LLMPort":
        """Get LLM with structured output."""
        ...


class EmbeddingsPort(Protocol):
    """Port for embeddings operations."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for documents."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query."""
        ...
