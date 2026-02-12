import logging

from fastapi import APIRouter, UploadFile

from src.api.schemas import BatchResponseSchema, DocumentResultSchema, MetadataSchema
from src.application.dto.batch_upload_dto import BatchProcessDocumentRequest
from src.application.dto.upload_dto import ProcessDocumentRequest
from src.container import get_container

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/batch", response_model=BatchResponseSchema)
async def batch_upload(files: list[UploadFile]) -> BatchResponseSchema:
    """Process multiple uploaded documents."""
    container = get_container()

    documents = []
    for file in files:
        content = await file.read()
        documents.append(
            ProcessDocumentRequest(
                content=content,
                filename=file.filename or "unknown",
                content_type=file.content_type or "application/octet-stream",
            )
        )

    request = BatchProcessDocumentRequest(documents=documents)
    response = container.batch_process_document_use_case.execute(request)

    results = [
        DocumentResultSchema(
            filename=r.filename,
            success=r.response.success,
            chunks_created=r.response.chunks_created,
            message=r.response.message,
            metadata=MetadataSchema(**r.response.metadata.model_dump()),
        )
        for r in response.results
    ]

    return BatchResponseSchema(
        total_documents=response.total_documents,
        successful=response.successful,
        failed=response.failed,
        results=results,
    )
