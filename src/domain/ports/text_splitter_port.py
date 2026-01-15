from typing import Protocol


class TextSplitterPort(Protocol):
    """Port for text splitting operations."""

    def split(self, text: str) -> list[str]:
        """Split text into chunks."""
        ...
