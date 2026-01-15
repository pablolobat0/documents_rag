from langchain_text_splitters import RecursiveCharacterTextSplitter


class LangchainTextSplitter:
    """LangChain text splitter implementation. Implements TextSplitterPort."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def split(self, text: str) -> list[str]:
        """Split text into chunks."""
        return self._splitter.split_text(text)
