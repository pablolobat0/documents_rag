from langchain_core.messages import AIMessage, HumanMessage
from app.schemas.chat import ChatConversation
from src.agent import Agent


class ChatService:
    def __init__(self, agent: Agent) -> None:
        self.agent = agent

    def chat(self, conversation: ChatConversation) -> str:
        session_id = conversation.session_id
        messages = conversation.model_dump()["messages"]
        return self.agent.run(
            {
                "messages": [
                    (
                        HumanMessage(content=message["content"])
                        if message["role"] == "user"
                        else AIMessage(content=message["content"])
                    )
                    for message in messages
                ]
            },
            session_id,
        )