import streamlit as st

from config.settings import settings
from src.application.dto.upload_dto import ProcessDocumentRequest
from src.application.use_cases.process_document import ProcessDocumentUseCase
from src.presentation.state.session_state import increment_document_count


def render_file_uploader(process_use_case: ProcessDocumentUseCase):
    """Render the file upload interface."""
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["txt", "pdf", "md"],
        accept_multiple_files=False,
        help="Supported formats: TXT, PDF, Markdown (max 10MB)",
    )

    if uploaded_file is not None:
        # Validate file size
        if uploaded_file.size > settings.max_file_size:
            st.error("File too large. Maximum size is 10MB.")
            return

        if st.button("Process Document", type="primary", use_container_width=True):
            with st.spinner("Processing document..."):
                try:
                    request = ProcessDocumentRequest(
                        content=uploaded_file.getvalue(),
                        filename=uploaded_file.name,
                        content_type=uploaded_file.type,
                    )
                    response = process_use_case.execute(request)

                    if response.success:
                        st.success(
                            f"Document processed! Created {response.chunks_created} chunks."
                        )
                        increment_document_count()

                        # Show metadata in expander
                        with st.expander("View Metadata"):
                            metadata_dict = response.metadata.model_dump()
                            # Filter out None values for cleaner display
                            filtered_metadata = {
                                k: v for k, v in metadata_dict.items() if v is not None
                            }
                            st.json(filtered_metadata)
                    else:
                        st.error(f"Failed: {response.message}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
