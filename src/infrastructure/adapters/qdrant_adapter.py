import uuid

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams

from src.infrastructure.adapters.ollama_adapter import OllamaAdapter


class QdrantAdapter:
    """Adapter for Qdrant vector store."""

    def __init__(
        self,
        url: str,
        collection_name: str,
        ollama_adapter: OllamaAdapter,
    ):
        self.url = url
        self.collection_name = collection_name
        self.client = QdrantClient(url=url)
        self.ollama_adapter = ollama_adapter

        # Get vector size from embeddings
        embeddings = ollama_adapter.get_embeddings()
        vector_size = len(embeddings.embed_query("sample text"))

        # Ensure collection exists
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

        # Create LangChain vector store for retrieval
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=embeddings,
        )

    def get_retriever(self, search_type: str = "mmr", k: int = 10):
        """Get a retriever for searching documents."""
        return self.vector_store.as_retriever(
            search_type=search_type, search_kwargs={"k": k}
        )

    def get_vector_store(self) -> QdrantVectorStore:
        """Get the underlying vector store."""
        return self.vector_store

    def upsert(
        self, chunks: list[str], embeddings: list[list[float]], metadata: dict | None = None
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
