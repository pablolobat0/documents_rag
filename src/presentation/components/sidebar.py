import streamlit as st

from config.settings import settings
from src.presentation.state.session_state import reset_session


def render_sidebar():
    """Render the sidebar with session info and settings."""
    st.header("Session")

    # Session ID display
    st.text_input(
        "Session ID",
        value=st.session_state.session_id[:8] + "...",
        disabled=True,
        help=f"Full ID: {st.session_state.session_id}",
    )

    # New session button
    if st.button("New Session", use_container_width=True):
        reset_session()
        st.rerun()

    st.divider()

    # Document stats
    st.header("Documents")
    st.metric("Uploaded", st.session_state.get("document_count", 0))

    st.divider()

    # Settings info
    st.header("Settings")
    st.caption(f"Model: {settings.model}")
    st.caption(f"Embeddings: {settings.embeddings_model}")
