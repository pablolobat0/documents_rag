from src.application.dto.chat_dto import ChatRequest, ChatResponse
from src.domain.ports.agent_port import AgentPort


class ChatWithDocumentsUseCase:
    """Use case for chatting with documents using RAG."""

    def __init__(self, agent: AgentPort):
        self._agent = agent

    def execute(self, request: ChatRequest) -> ChatResponse:
        """Execute the chat use case."""
        response_content = self._agent.run(
            messages=request.messages,
            session_id=str(request.session_id),
        )

        return ChatResponse(
            content=response_content,
            session_id=str(request.session_id),
        )
