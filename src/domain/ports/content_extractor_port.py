from typing import Protocol

from src.domain.value_objects.page_content import PageContent


class ContentExtractorPort(Protocol):
    """Port for extracting content from various document formats."""

    def extract_content(self, file_content: bytes) -> tuple[list[PageContent], int]:
        """
        Extract content from a document.

        Args:
            file_content: Raw file bytes

        Returns:
            Tuple of (list of PageContent with section/page numbers, total sections/pages)
        """
        ...

    @property
    def supported_content_types(self) -> list[str]:
        """Return list of MIME types this extractor supports."""
        ...
