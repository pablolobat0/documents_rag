from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DocumentChunk:
    content: str
    embedding: list[float] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Document:
    id: str
    name: str
    content: str
    file_type: str
    file_size: int
    chunks: list[DocumentChunk] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: datetime | None = None

    def add_chunk(self, chunk: DocumentChunk) -> None:
        self.chunks.append(chunk)
