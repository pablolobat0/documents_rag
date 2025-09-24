from datetime import datetime
from fastapi import UploadFile, File, APIRouter, Depends
from fastapi.responses import JSONResponse

from app.services.vector_storage import VectorStorageService
from app.dependencies import get_vector_storage_service

vector_storage_router = APIRouter(prefix="/document", tags=["documents"])


@vector_storage_router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    vector_storage_service: VectorStorageService = Depends(get_vector_storage_service),
):
    content = await file.read()

    # Extract file information for metadata
    file_info = {
        "document_name": file.filename,
        "file_path": None,
        "file_size": len(content),  # Size in bytes
        "file_type": file.content_type,
        "created_at": datetime.now(),
    }

    if file.content_type == "text/plain":
        vector_storage_service.insert_documents(
            documents_text=[content.decode()], file_info=file_info
        )
    elif file.content_type == "application/pdf":
        vector_storage_service.insert_pdf_document(
            file_content=content, file_info=file_info
        )
    else:
        return JSONResponse({"error": "Unsupported file type"}, status_code=400)

    return JSONResponse({"message": "Documento insertado con éxito"})
