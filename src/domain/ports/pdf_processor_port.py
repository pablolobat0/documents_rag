from typing import Protocol


class PdfProcessorPort(Protocol):
    """Port for PDF processing operations."""

    def extract_content(self, file_content: bytes) -> tuple[list[str], int]:
        """
        Extract content from a PDF file.

        Args:
            file_content: Raw PDF file bytes

        Returns:
            Tuple of (list of text content, number of pages)
        """
        ...
