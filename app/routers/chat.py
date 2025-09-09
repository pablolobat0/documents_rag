from fastapi import APIRouter, Depends, status

from app.schemas.chat import ChatConversation
from app.dependencies import get_chat_service
from app.services.chat import ChatService

chat_router = APIRouter(prefix="/chat", tags=["chat"])


@chat_router.post(
    "/",
    status_code=status.HTTP_200_OK,
)
async def chat(
    conversation: ChatConversation,
    chat_service: ChatService = Depends(get_chat_service),
) -> str:
    return chat_service.chat(conversation)
