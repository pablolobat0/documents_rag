from dataclasses import dataclass

from src.domain.entities.metadata import Metadata


@dataclass
class ProcessDocumentRequest:
    content: bytes
    filename: str
    content_type: str


@dataclass
class ProcessDocumentResponse:
    success: bool
    metadata: Metadata
    chunks_created: int
    message: str = ""
