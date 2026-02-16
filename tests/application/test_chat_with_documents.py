from unittest.mock import MagicMock

from src.application.dto.chat_dto import ChatRequest
from src.application.use_cases.chat_with_documents import ChatWithDocumentsUseCase
from src.domain.value_objects.chat_message import ChatMessage, SessionId


def _make_request() -> ChatRequest:
    return ChatRequest(
        session_id=SessionId(value="session-1"),
        messages=[ChatMessage(role="user", content="Hello")],
    )


class TestExecute:
    def test_returns_agent_response(self):
        agent = MagicMock()
        agent.run.return_value = "Answer from agent"
        uc = ChatWithDocumentsUseCase(agent=agent)

        response = uc.execute(_make_request())

        assert response.content == "Answer from agent"
        assert response.session_id == "session-1"
        agent.run.assert_called_once()
