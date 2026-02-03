from typing import Protocol

from src.domain.value_objects.page_content import PageContent


class PdfProcessorPort(Protocol):
    """Port for PDF processing operations."""

    def extract_content(self, file_content: bytes) -> tuple[list[PageContent], int]:
        """
        Extract content from a PDF file.

        Args:
            file_content: Raw PDF file bytes

        Returns:
            Tuple of (list of PageContent with page numbers, number of pages)
        """
        ...
