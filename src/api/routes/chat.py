import logging

from fastapi import APIRouter, HTTPException

from src.api.schemas import ChatRequestSchema, ChatResponseSchema
from src.application.dto.chat_dto import ChatRequest
from src.container import get_container
from src.domain.value_objects.chat_message import ChatMessage, SessionId

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


@router.post(
    "/chat",
    response_model=ChatResponseSchema,
    summary="Chat with documents",
    description="Send the full conversation history and receive an AI-generated "
    "response grounded in the indexed documents. The session_id maintains "
    "conversation state across requests.",
)
async def chat(request: ChatRequestSchema) -> ChatResponseSchema:
    container = get_container()

    try:
        chat_request = ChatRequest(
            session_id=SessionId(request.session_id),
            messages=[
                ChatMessage(role=m.role, content=m.content) for m in request.messages
            ],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        response = container.chat_use_case.execute(chat_request)
    except Exception as exc:
        logger.exception("Chat execution failed for session %s", request.session_id)
        raise HTTPException(
            status_code=500, detail="Internal error during chat processing"
        ) from exc

    return ChatResponseSchema(
        content=response.content,
        session_id=response.session_id,
        timestamp=response.timestamp,
        sources_used=response.sources_used,
        tool_calls=response.tool_calls,
    )
