from datetime import datetime
from fastapi import UploadFile, File, APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
import logging

from app.services.vector_storage import VectorStorageService
from app.dependencies import get_vector_storage_service

vector_storage_router = APIRouter(prefix="/document", tags=["documents"])
logger = logging.getLogger(__name__)


@vector_storage_router.post(
    "/upload",
    summary="Upload document for processing",
    description="Upload a text or PDF document to be processed and added to the knowledge base. The document will be chunked, embedded, and made available for chat queries.",
)
async def upload_document(
    file: UploadFile = File(...),
    vector_storage_service: VectorStorageService = Depends(get_vector_storage_service),
):
    try:
        logger.info(f"File upload request received: {file.filename}, size: {file.size}, type: {file.content_type}")

        # Validate file size (10MB limit)
        max_file_size = 10 * 1024 * 1024  # 10MB
        if file.size and file.size > max_file_size:
            logger.warning(f"File too large: {file.filename}, size: {file.size}")
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {max_file_size//1024//1024}MB"
            )

        content = await file.read()

        # Validate file was successfully read
        if not content:
            logger.warning(f"Empty file received: {file.filename}")
            raise HTTPException(status_code=400, detail="Empty file")

        # Extract file information for metadata
        file_info = {
            "document_name": file.filename or "unknown",
            "file_path": None,
            "file_size": len(content),  # Size in bytes
            "file_type": file.content_type,
            "created_at": datetime.now(),
        }

        # Process based on file type
        if file.content_type == "text/plain":
            try:
                text_content = content.decode('utf-8')
                vector_storage_service.insert_documents(
                    documents_text=[text_content], file_info=file_info
                )
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="Invalid text encoding. Use UTF-8.")
        elif file.content_type == "application/pdf":
            vector_storage_service.insert_pdf_document(
                file_content=content, file_info=file_info
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only text/plain and application/pdf are supported.")

        logger.info(f"Successfully processed document: {file.filename}")
        return JSONResponse({"message": "Documento insertado con éxito"})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload for {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while processing document")
