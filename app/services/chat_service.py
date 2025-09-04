from app.schemas.chat import ChatConversation
from src.agent import Agent


class ChatService:

    def __init__(self) -> None:
        self.agent = Agent()

    def chat(self, conversation: ChatConversation) -> str:
        return self.agent.run(conversation.model_dump())
