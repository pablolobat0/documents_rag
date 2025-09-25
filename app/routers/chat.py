from datetime import datetime
from fastapi import APIRouter, Depends, status
import logging

from app.schemas.chat import ChatConversation, ChatResponse
from app.dependencies import get_chat_service
from app.services.chat import ChatService

chat_router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


@chat_router.post(
    "/",
    status_code=status.HTTP_200_OK,
    response_model=ChatResponse,
    summary="Chat with uploaded documents",
    description="Send a message to the AI assistant and receive a response based on the uploaded documents. The system maintains conversation history per session.",
)
async def chat(
    conversation: ChatConversation,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    logger.info(f"Chat request received for session {conversation.session_id}")

    try:
        response_text = chat_service.chat(conversation)
        logger.info(f"Chat response generated for session {conversation.session_id}")

        return ChatResponse(
            response=response_text,
            session_id=conversation.session_id,
            timestamp=datetime.now().isoformat(),
            sources_used=[],  # TODO: Implement source tracking from tool calls
            tool_calls=None   # TODO: Extract tool call information
        )
    except Exception as e:
        logger.error(f"Error processing chat request for session {conversation.session_id}: {str(e)}")
        raise
