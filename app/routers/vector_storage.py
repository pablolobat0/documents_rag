from fastapi import UploadFile, File, APIRouter, Depends
from fastapi.responses import JSONResponse

from app.services.vector_storage import VectorStorageService
from app.dependencies import get_vector_storage_service

vector_storage_router = APIRouter(prefix="/chat", tags=["chat"])


@vector_storage_router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    vector_storage_service: VectorStorageService = Depends(get_vector_storage_service),
):
    try:
        # Leer el contenido del archivo
        content = await file.read()
        text = content.decode("utf-8")

        vector_storage_service.insert_document(text)

        return JSONResponse({"message": "Documento insertado con éxito"})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
