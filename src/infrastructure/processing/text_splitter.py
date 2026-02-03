from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.domain.value_objects.document_chunk import DocumentChunk
from src.domain.value_objects.page_content import PageContent


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

    def split_pages(
        self, pages: list[PageContent], base_metadata: dict
    ) -> list[DocumentChunk]:
        """
        Split pages into chunks while preserving page numbers.

        Each chunk inherits its source page's metadata.
        """
        texts = []
        metadatas = []

        for page in pages:
            page_metadata = {
                **base_metadata,
                "page_number": page.page_number,
                "content_type": page.content_type,
            }
            texts.append(page.content)
            metadatas.append(page_metadata)

        # Use create_documents to split while preserving metadata
        langchain_docs = self._splitter.create_documents(texts, metadatas)

        return [
            DocumentChunk(
                content=doc.page_content,
                metadata=doc.metadata,
            )
            for doc in langchain_docs
        ]
