import uuid

import streamlit as st


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "document_count" not in st.session_state:
        st.session_state.document_count = 0


def reset_session():
    """Reset the session to a new state."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []


def add_message(role: str, content: str):
    """Add a message to the session state."""
    st.session_state.messages.append({"role": role, "content": content})


def increment_document_count():
    """Increment the document count."""
    st.session_state.document_count += 1
