import streamlit as st

from src.container import get_container
from src.presentation.components.batch_file_uploader import render_batch_file_uploader
from src.presentation.components.chat_interface import render_chat
from src.presentation.components.sidebar import render_sidebar
from src.presentation.state.session_state import initialize_session_state


def main():
    st.set_page_config(
        page_title="Documents RAG",
        page_icon="📄",
        layout="wide",
    )

    # Initialize session state
    initialize_session_state()

    # Get container with all dependencies
    container = get_container()

    # Title
    st.title("📄 Documents RAG")

    # Sidebar
    with st.sidebar:
        render_sidebar()
        st.divider()
        st.subheader("📤 Upload")
        render_batch_file_uploader(container.batch_process_document_use_case)

    # Main chat area
    render_chat(container.chat_use_case)


if __name__ == "__main__":
    main()
