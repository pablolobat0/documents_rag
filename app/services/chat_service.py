from langchain_core.messages import AIMessage, HumanMessage
from app.schemas.chat import ChatConversation
from src.agent import Agent


class ChatService:

    def __init__(self) -> None:
        self.agents: dict[str, Agent] = {}

    def chat(self, conversation: ChatConversation) -> str:
        session_id = conversation.session_id
        if session_id not in self.agents:
            self.agents[session_id] = Agent()

        agent = self.agents[session_id]

        messages = conversation.model_dump()["messages"]
        return agent.run(
            {
                "messages": [
                    (
                        HumanMessage(content=message["content"])
                        if message["role"] == "user"
                        else AIMessage(content=message["content"])
                    )
                    for message in messages
                ]
            }
        )
