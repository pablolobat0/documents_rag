from langchain_core.messages import AIMessage, HumanMessage

from src.application.dto.chat_dto import ChatRequest, ChatResponse
from src.infrastructure.rag.agent import Agent


class ChatWithDocumentsUseCase:
    """Use case for chatting with documents using RAG."""

    def __init__(self, agent: Agent):
        self.agent = agent

    def execute(self, request: ChatRequest) -> ChatResponse:
        """Execute the chat use case."""
        # Convert domain messages to LangChain format
        langchain_messages = []
        for msg in request.messages:
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            else:
                langchain_messages.append(AIMessage(content=msg.content))

        # Run the agent
        response_content = self.agent.run(
            {"messages": langchain_messages},
            session_id=str(request.session_id),
        )

        return ChatResponse(
            content=response_content,
            session_id=str(request.session_id),
        )
