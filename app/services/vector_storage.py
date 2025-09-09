import uuid
import tempfile
import os
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from qdrant_client.models import Distance, VectorParams
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.http import models
from langchain_community.document_loaders import PyPDFLoader


class VectorStorageService:

    def __init__(self, url: str, embeddings_model: str, embeddings_url: str) -> None:
        self.client = QdrantClient(url=url)
        self.embeddings = OllamaEmbeddings(
            model=embeddings_model, base_url=embeddings_url
        )

        vector_size = len(self.embeddings.embed_query("sample text"))

        if not self.client.collection_exists("test"):
            self.client.create_collection(
                collection_name="test",
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def insert_document(self, document_text: str, metadata: dict | None = None) -> None:
        # Split in chunks for an improved retrieval
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.split_text(document_text)
        if not chunks:
            raise ValueError("El texto proporcionado está vacío")

        # Generate the embeddings for each chunk
        embeddings = self.embeddings.embed_documents(chunks)

        # Insert in vector database
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

        self.client.upsert(collection_name="test", points=points)

    def insert_pdf_document(
        self, file_content: bytes, metadata: dict | None = None
    ) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            pages = []
            for page in loader.load():
                pages.append(page.page_content)

            self.insert_document(document_text="\n".join(pages), metadata=metadata)
        finally:
            os.remove(tmp_path)
