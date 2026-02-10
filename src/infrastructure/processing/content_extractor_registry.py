from src.domain.ports.content_extractor_port import ContentExtractorPort


class ContentExtractorRegistry:
    """Registry for content extractors by MIME type."""

    def __init__(self):
        self._extractors: dict[str, ContentExtractorPort] = {}

    def register(self, extractor: ContentExtractorPort) -> None:
        """Register an extractor for its supported content types."""
        for content_type in extractor.supported_content_types:
            self._extractors[content_type] = extractor

    def get_extractor(self, content_type: str) -> ContentExtractorPort | None:
        """Get the appropriate extractor for a content type."""
        return self._extractors.get(content_type)

    def supports(self, content_type: str) -> bool:
        """Check if a content type is supported."""
        return content_type in self._extractors

    @property
    def supported_types(self) -> list[str]:
        """Get all supported content types."""
        return list(self._extractors.keys())
