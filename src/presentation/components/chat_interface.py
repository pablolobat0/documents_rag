import streamlit as st

from src.application.dto.chat_dto import ChatRequest
from src.application.use_cases.chat_with_documents import ChatWithDocumentsUseCase
from src.domain.value_objects.chat_message import ChatMessage, SessionId
from src.presentation.state.session_state import add_message


def render_chat(chat_use_case: ChatWithDocumentsUseCase):
    """Render the chat interface."""
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        # Add user message
        add_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    request = ChatRequest(
                        session_id=SessionId(st.session_state.session_id),
                        messages=[
                            ChatMessage(role=m["role"], content=m["content"])
                            for m in st.session_state.messages
                        ],
                    )
                    response = chat_use_case.execute(request)
                    st.markdown(response.content)
                    add_message("assistant", response.content)
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    add_message("assistant", error_msg)
