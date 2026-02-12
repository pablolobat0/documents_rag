from unittest.mock import MagicMock

import pytest

from src.domain.ports.content_extractor_port import ContentExtractorPort
from src.infrastructure.processing.content_extractor_registry import (
    ContentExtractorRegistry,
)


@pytest.fixture
def registry():
    return ContentExtractorRegistry()


def _make_extractor(content_types: list[str]) -> MagicMock:
    extractor = MagicMock(spec=ContentExtractorPort)
    extractor.supported_content_types = content_types
    return extractor


class TestContentExtractorRegistry:
    def test_register_and_retrieve(self, registry):
        extractor = _make_extractor(["text/plain"])
        registry.register(extractor)
        assert registry.get_extractor("text/plain") is extractor

    def test_unsupported_type_returns_none(self, registry):
        assert registry.get_extractor("application/xyz") is None

    def test_supports_bool_check(self, registry):
        extractor = _make_extractor(["text/plain"])
        registry.register(extractor)
        assert registry.supports("text/plain") is True
        assert registry.supports("text/html") is False

    def test_multiple_mime_types_per_extractor(self, registry):
        extractor = _make_extractor(["text/markdown", "text/x-markdown"])
        registry.register(extractor)
        assert registry.get_extractor("text/markdown") is extractor
        assert registry.get_extractor("text/x-markdown") is extractor

    def test_supported_types_lists_all(self, registry):
        registry.register(_make_extractor(["text/plain"]))
        registry.register(_make_extractor(["application/pdf"]))
        types = registry.supported_types
        assert "text/plain" in types
        assert "application/pdf" in types

    def test_later_registration_overwrites(self, registry):
        first = _make_extractor(["text/plain"])
        second = _make_extractor(["text/plain"])
        registry.register(first)
        registry.register(second)
        assert registry.get_extractor("text/plain") is second
