from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedDocument:
    """A document retrieved from the vector store."""

    page_content: str
    metadata: dict | None = None
