import logging

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore as LangchainQdrantVectorStore
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams, SparseVectorParams


from src.domain.value_objects.document_chunk import DocumentChunk
from src.domain.value_objects.retrieved_document import RetrievedDocument

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Qdrant vector store implementation. Implements VectorStorePort."""

    def __init__(
        self,
        url: str,
        collection_name: str,
        embeddings: Embeddings,
    ):
        self.url = url
        self.collection_name = collection_name
        self.client = QdrantClient(url=url)
        self._embeddings = embeddings

        vector_size = len(embeddings.embed_query("sample text"))

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={"dense": VectorParams(size=vector_size, distance=Distance.COSINE)},
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )

        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        self._vector_store = LangchainQdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self._embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Insert or update document chunks with per-chunk metadata."""
        if not chunks:
            return

        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        self._vector_store.add_texts(texts=texts, metadatas=metadatas)

    def search(self, query: str, num_documents: int) -> list[RetrievedDocument]:
        """Search for relevant documents and return domain objects."""
        if not query or not query.strip():
            logger.warning("Empty query provided for search")
            return []

        docs = self._vector_store.similarity_search(query, k=num_documents)
        return [
            RetrievedDocument(
                page_content=doc.page_content,
                metadata=doc.metadata,
            )
            for doc in docs
        ]
