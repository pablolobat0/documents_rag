import uuid
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from qdrant_client.models import Distance, VectorParams
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.http import models
from app.services.image_captioning import ImageCaptioningService
import pypdf
import io


class VectorStorageService:

    def __init__(
        self,
        url: str,
        embeddings_model: str,
        embeddings_url: str,
        image_captioning_service: ImageCaptioningService,
    ) -> None:
        self.client = QdrantClient(url=url)
        self.embeddings = OllamaEmbeddings(
            model=embeddings_model, base_url=embeddings_url
        )
        self.image_captioning_service = image_captioning_service

        vector_size = len(self.embeddings.embed_query("sample text"))

        if not self.client.collection_exists("test"):
            self.client.create_collection(
                collection_name="test",
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def insert_documents(
        self,
        documents_text: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        # Split in chunks for an improved retrieval
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.split_text("\n".join(documents_text))
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
                        **(metadatas or {}),
                    },
                )
            )

        self.client.upsert(collection_name="test", points=points)

    def insert_pdf_document(self, file_content: bytes) -> None:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = pypdf.PdfReader(pdf_file)

        documents = []
        metadatas = []

        for page in pdf_reader.pages:
            documents.append(page.extract_text())
            for image in page.images:
                image_summary = self.image_captioning_service.get_image_summary(
                    image.data
                )
                documents.append(image_summary)

        self.insert_documents(documents_text=documents, metadatas=metadatas)
