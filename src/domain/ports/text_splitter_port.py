from typing import Protocol

from src.domain.value_objects.document_chunk import DocumentChunk
from src.domain.value_objects.page_content import PageContent


class TextSplitterPort(Protocol):
    """Port for text splitting operations."""

    def split_pages(
        self, pages: list[PageContent], base_metadata: dict
    ) -> list[DocumentChunk]:
        """Split pages into chunks, preserving page metadata for each chunk."""
        ...
