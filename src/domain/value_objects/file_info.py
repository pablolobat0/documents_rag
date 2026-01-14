from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class FileInfo:
    filename: str
    content_type: str
    size: int
    created_at: datetime | None = None

    def __post_init__(self):
        if self.size < 0:
            raise ValueError("File size cannot be negative")
        if not self.filename:
            raise ValueError("Filename cannot be empty")
