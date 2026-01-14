from dataclasses import dataclass
from typing import Union

from src.domain.entities.metadata import CurriculumVitae, Metadata, Receipt


@dataclass
class ProcessDocumentRequest:
    content: bytes
    filename: str
    content_type: str


@dataclass
class ProcessDocumentResponse:
    success: bool
    metadata: Union[Metadata, CurriculumVitae, Receipt]
    chunks_created: int
    message: str = ""
