from datetime import datetime

from pydantic import BaseModel, Field


class Metadata(BaseModel):
    """Base metadata for documents."""

    pages: int
    document_name: str | None = None
    file_path: str | None = None
    file_size: int | None = None
    file_type: str | None = None
    created_at: datetime | None = None
    processed_at: datetime | None = None
    frontmatter: dict = Field(default_factory=dict)
