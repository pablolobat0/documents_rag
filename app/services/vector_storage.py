import uuid
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from qdrant_client.models import Distance, VectorParams
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.http import models
from app.services.image_captioning import ImageCaptioningService
import pypdf
import io
from datetime import datetime
from pathlib import Path

from app.services.metadata import MetadataService
from app.schemas.metadata import Metadata


class VectorStorageService:

    def __init__(
        self,
        url: str,
        embeddings_model: str,
        embeddings_url: str,
        image_captioning_service: ImageCaptioningService,
        metadata_service: MetadataService,
    ) -> None:
        self.client = QdrantClient(url=url)
        self.embeddings = OllamaEmbeddings(
            model=embeddings_model, base_url=embeddings_url
        )
        self.image_captioning_service = image_captioning_service
        self.metadata_service = metadata_service

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
        file_info: dict | None = None,
    ) -> None:
        # Create base metadata with file information
        # For text files, we count each document as a "page"
        pages = len(documents_text)
        base_metadata = self._create_base_metadata(file_info, pages)

        # Extract structured metadata using the base metadata
        extracted_metadata = self.metadata_service.extract_metadata(
            "\n".join(documents_text), base_metadata
        )

        # Save metadata to JSON file
        try:
            self.metadata_service.storage_service.save_metadata(extracted_metadata)
        except Exception as e:
            print(f"Warning: Failed to save metadata: {e}")

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

    def _create_base_metadata(self, file_info: dict | None, pages: int) -> Metadata:
        """
        Create a base Metadata object with file information.
        """
        if not file_info:
            return Metadata(pages=pages)

        # Use PDF page count if available, otherwise use calculated pages
        actual_pages = file_info.get("pages", pages)

        return Metadata(
            pages=actual_pages,
            document_name=file_info.get("document_name"),
            file_path=file_info.get("file_path"),
            file_size=file_info.get("file_size"),
            file_type=file_info.get("file_type"),
            created_at=file_info.get("created_at"),
            processed_at=datetime.now(),
        )

    def insert_pdf_document(
        self, file_content: bytes, file_info: dict | None = None
    ) -> None:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = pypdf.PdfReader(pdf_file)

        documents = []
        # TODO: add document name and page number in metadata for sources
        metadatas = []

        # Count actual PDF pages
        pdf_pages = len(pdf_reader.pages)

        for page in pdf_reader.pages:
            documents.append(page.extract_text())
            for image in page.images:
                image_summary = self.image_captioning_service.get_image_summary(
                    image.data
                )
                documents.append(image_summary)

        # Update file_info with correct page count
        if file_info:
            file_info["pages"] = pdf_pages

        self.insert_documents(
            documents_text=documents, metadatas=metadatas, file_info=file_info
        )
