from unittest.mock import MagicMock, create_autospec

import pytest

from src.domain.ports.content_extractor_port import ContentExtractorPort
from src.domain.ports.text_splitter_port import TextSplitterPort
from src.domain.ports.vector_store_port import VectorStorePort
from src.domain.value_objects.document_chunk import DocumentChunk
from src.domain.value_objects.extraction_result import ExtractionResult
from src.domain.value_objects.page_content import PageContent
from src.infrastructure.processing.content_extractor_registry import (
    ContentExtractorRegistry,
)

# --- Markdown sample fixtures ---


@pytest.fixture
def markdown_with_frontmatter() -> bytes:
    return b"""---
title: Test Document
author: John Doe
tags:
  - python
  - testing
---

# Introduction

This is the introduction.

## Details

Some details here.
"""


@pytest.fixture
def markdown_without_frontmatter() -> bytes:
    return b"""# Introduction

This is the introduction.

## Details

Some details here.
"""


@pytest.fixture
def markdown_no_headers() -> bytes:
    return b"Just some plain text without any headers."


# --- PageContent helpers ---


@pytest.fixture
def sample_pages() -> list[PageContent]:
    return [
        PageContent(content="Page one content.", page_number=1, content_type="text"),
        PageContent(content="Page two content.", page_number=2, content_type="text"),
    ]


@pytest.fixture
def single_page() -> list[PageContent]:
    return [
        PageContent(content="Single page content.", page_number=1, content_type="text"),
    ]


# --- Mock ports ---


@pytest.fixture
def mock_vector_store():
    return create_autospec(VectorStorePort, instance=True)


@pytest.fixture
def mock_text_splitter():
    splitter = create_autospec(TextSplitterPort, instance=True)
    splitter.split_pages.return_value = [
        DocumentChunk(content="chunk 1", metadata={"page_number": 1}),
        DocumentChunk(content="chunk 2", metadata={"page_number": 2}),
    ]
    return splitter


@pytest.fixture
def mock_registry_with_extractor():
    """Real ContentExtractorRegistry with a mock extractor for text/plain."""
    registry = ContentExtractorRegistry()
    mock_extractor = MagicMock(spec=ContentExtractorPort)
    mock_extractor.supported_content_types = ["text/plain"]
    mock_extractor.extract_content.return_value = ExtractionResult(
        page_contents=[
            PageContent(content="extracted text", page_number=1, content_type="text"),
        ],
        total_pages=1,
    )
    registry.register(mock_extractor)
    return registry
