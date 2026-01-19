from typing import Any
import uuid

from langchain_core.embeddings import Embeddings
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
        embeddings: list[list[float]],
        metadata: dict | None = None,
    ) -> None:
        """Insert or update document chunks."""
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
    ) -> list[Any]:
        """Get a retriever for searching documents."""
        return self._retriever.invoke(query)
