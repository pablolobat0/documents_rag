from typing import Protocol

from src.domain.value_objects.extraction_result import ExtractionResult


class ContentExtractorPort(Protocol):
    """Port for extracting content from various document formats."""

    def extract_content(self, file_content: bytes) -> ExtractionResult:
        """
        Extract content from a document.

        Args:
            file_content: Raw file bytes

        Returns:
            ExtractionResult with page contents, total pages, and document metadata
        """
        ...

    @property
    def supported_content_types(self) -> list[str]:
        """Return list of MIME types this extractor supports."""
        ...
