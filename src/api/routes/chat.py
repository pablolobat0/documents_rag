import logging

from fastapi import APIRouter

from src.api.schemas import ChatRequestSchema, ChatResponseSchema
from src.application.dto.chat_dto import ChatRequest
from src.container import get_container
from src.domain.value_objects.chat_message import ChatMessage, SessionId

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponseSchema)
async def chat(request: ChatRequestSchema) -> ChatResponseSchema:
    """Send a chat message and get a response."""
    container = get_container()

    chat_request = ChatRequest(
        session_id=SessionId(request.session_id),
        messages=[
            ChatMessage(role=m.role, content=m.content) for m in request.messages
        ],
    )

    response = container.chat_use_case.execute(chat_request)

    return ChatResponseSchema(
        content=response.content,
        session_id=response.session_id,
        timestamp=response.timestamp,
        sources_used=response.sources_used,
        tool_calls=response.tool_calls,
    )
