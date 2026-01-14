from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings


class TextSplitter:
    """Domain service for splitting text into chunks."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def split(self, text: str) -> list[str]:
        """Split text into chunks."""
        return self._splitter.split_text(text)
