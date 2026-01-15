from typing import Any, Protocol, Union

from src.domain.value_objects.chat_message import ChatMessage


class LLMPort(Protocol):
    """Port for LLM operations."""

    def invoke(self, messages: list[ChatMessage]) -> str:
        """Invoke LLM with conversation messages and return response content."""
        ...

    def invoke_structured(
        self,
        messages: Union[list[ChatMessage], list[dict]],
        schema: type,
    ) -> Any:
        """Invoke LLM with messages and parse response into structured schema."""
        ...


class EmbeddingsPort(Protocol):
    """Port for embeddings operations."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for documents."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query."""
        ...
