from typing import Protocol, Union

from src.domain.entities.metadata import CurriculumVitae, Metadata, Receipt
from src.domain.ports.document_classifier_port import DocumentType


class MetadataExtractorPort(Protocol):
    """Port for metadata extraction."""

    def extract(
        self, text: str, document_type: DocumentType, base_metadata: Metadata
    ) -> Union[Metadata, CurriculumVitae, Receipt]:
        """Extract metadata from document based on its type."""
        ...
