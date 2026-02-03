from dataclasses import dataclass


@dataclass(frozen=True)
class PageContent:
    """Represents content extracted from a single page."""

    content: str
    page_number: int  # 1-indexed
    content_type: str = "text"  # "text" or "image_caption"
