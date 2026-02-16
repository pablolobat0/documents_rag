import pytest

from src.domain.value_objects.chat_message import ChatMessage, SessionId


class TestChatMessage:
    def test_valid_user_role(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_valid_assistant_role(self):
        msg = ChatMessage(role="assistant", content="Hi there")
        assert msg.role == "assistant"

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError, match="Invalid role"):
            ChatMessage(role="system", content="test")

    def test_empty_content_allowed(self):
        msg = ChatMessage(role="user", content="")
        assert msg.content == ""


class TestSessionId:
    def test_valid_id(self):
        sid = SessionId(value="abc-123")
        assert str(sid) == "abc-123"

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            SessionId(value="")
