from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentChunk:
    """Represents a text chunk with its metadata."""

    content: str
    metadata: dict = field(default_factory=dict)
