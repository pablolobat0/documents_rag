import streamlit as st

from config.settings import settings
from src.application.dto.batch_upload_dto import (
    BatchProcessDocumentRequest,
    BatchProcessDocumentResponse,
)
from src.application.dto.upload_dto import ProcessDocumentRequest
from src.application.use_cases.batch_process_documents import (
    BatchProcessDocumentUseCase,
)
from src.presentation.state.session_state import increment_document_count

SUPPORTED_EXTENSIONS = ["txt", "pdf", "md"]


def render_batch_file_uploader(batch_use_case: BatchProcessDocumentUseCase):
    """Render the batch file upload interface with progress feedback."""
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        help=f"Supported formats: {', '.join(ext.upper() for ext in SUPPORTED_EXTENSIONS)} (max 10MB each)",
    )

    if uploaded_files:
        # Validate files
        valid_files = []
        for file in uploaded_files:
            if file.size > settings.max_file_size:
                st.warning(f"Skipping {file.name}: File too large (max 10MB)")
            else:
                valid_files.append(file)

        if valid_files:
            st.info(f"{len(valid_files)} file(s) ready for processing")

            if st.button(
                "Process All Documents", type="primary", use_container_width=True
            ):
                _process_batch(valid_files, batch_use_case)


def _process_batch(files: list, batch_use_case: BatchProcessDocumentUseCase):
    """Process a batch of files with progress display."""
    # Create progress container
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Build batch request
    documents = []
    for file in files:
        documents.append(
            ProcessDocumentRequest(
                content=file.getvalue(),
                filename=file.name,
                content_type=file.type or "application/octet-stream",
            )
        )

    request = BatchProcessDocumentRequest(documents=documents)

    def update_progress(current: int, total: int, filename: str):
        progress_bar.progress(current / total)
        status_text.text(f"Processing {current}/{total}: {filename}")

    # Execute batch processing
    response = batch_use_case.execute(request, progress_callback=update_progress)

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    # Display results
    _display_batch_results(response)


def _display_batch_results(response: BatchProcessDocumentResponse):
    """Display batch processing results."""
    if response.all_succeeded:
        st.success(
            f"All {response.total_documents} documents processed successfully! "
            f"Total chunks created: {response.total_chunks_created}"
        )
        for _ in range(response.successful):
            increment_document_count()
    else:
        st.warning(
            f"Processed {response.total_documents} documents: "
            f"{response.successful} succeeded, {response.failed} failed"
        )
        for _ in range(response.successful):
            increment_document_count()

    # Detailed results in expander
    with st.expander("View Processing Details"):
        for result in response.results:
            if result.response.success:
                st.success(
                    f"**{result.filename}**: {result.response.chunks_created} chunks created"
                )
            else:
                st.error(f"**{result.filename}**: {result.response.message}")
