import streamlit as st

from src.presentation.api_client import api_client
from src.presentation.state.session_state import add_message


def render_chat():
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
            try:
                with st.spinner("Thinking..."):
                    response = api_client.send_chat(
                        session_id=st.session_state.session_id,
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                    )
                content = response["content"]
                st.markdown(content)
                add_message("assistant", content)
            except Exception as e:
                error_msg = f"Error: {e!s}"
                st.error(error_msg)
                add_message("assistant", error_msg)
