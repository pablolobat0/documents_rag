from src.domain.value_objects.page_content import PageContent


class PlainTextProcessor:
    """Plain text processor implementation. Implements ContentExtractorPort."""

    @property
    def supported_content_types(self) -> list[str]:
        return ["text/plain"]

    def extract_content(self, file_content: bytes) -> tuple[list[PageContent], int]:
        """
        Extract content from a plain text file.

        Args:
            file_content: Raw file bytes

        Returns:
            Tuple of (list with single PageContent, 1)
        """
        content = file_content.decode("utf-8")
        return [PageContent(content=content, page_number=1, content_type="text")], 1
