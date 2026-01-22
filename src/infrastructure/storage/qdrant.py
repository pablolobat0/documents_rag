import logging
import uuid

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

from src.domain.value_objects.retrieved_document import RetrievedDocument
from langchain_qdrant import QdrantVectorStore as LangchainQdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams


class QdrantVectorStore:
    """Qdrant vector store implementation. Implements VectorStorePort."""

    def __init__(
        self,
        url: str,
        collection_name: str,
        search_type: str,
        n_results: int,
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
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

        self._vector_store = LangchainQdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self._embeddings,
        )

        self._retriever = self._vector_store.as_retriever(
            search_type=search_type, search_kwargs={"k": n_results}
        )

    def upsert(
        self,
        chunks: list[str],
        metadata: dict | None = None,
    ) -> None:
        """Insert or update document chunks."""
        embeddings = self._embeddings.embed_documents(chunks)
        points = []
        for chunk, vector in zip(chunks, embeddings):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "page_content": chunk,
                        **(metadata or {}),
                    },
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        query: str,
    ) -> list[RetrievedDocument]:
        """Search for relevant documents and return domain objects."""
        if not query or not query.strip():
            logger.warning("Empty query provided for search")
            return []

        docs = self._retriever.invoke(query)
        return [
            RetrievedDocument(
                page_content=doc.page_content,
                metadata=doc.metadata,
            )
            for doc in docs
        ]
