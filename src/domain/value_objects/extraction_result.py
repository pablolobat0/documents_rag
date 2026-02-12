from dataclasses import dataclass, field

from src.domain.value_objects.page_content import PageContent


@dataclass(frozen=True)
class ExtractionResult:
    """Result of extracting content from a document."""

    page_contents: list[PageContent]
    total_pages: int
    document_metadata: dict = field(default_factory=dict)
