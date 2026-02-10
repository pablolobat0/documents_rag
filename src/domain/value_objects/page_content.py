from dataclasses import dataclass


@dataclass(frozen=True)
class PageContent:
    """Represents content extracted from a single page or section."""

    content: str
    page_number: int  # 1-indexed, also used as section_number for non-paginated docs
    content_type: str = "text"  # "text" or "image_caption"
    section_title: str | None = None  # For markdown sections
