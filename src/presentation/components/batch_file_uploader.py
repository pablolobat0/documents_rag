import mimetypes
from pathlib import Path

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

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".md"}
EXTENSION_MIME_MAP = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".txt": "text/plain",
}


def render_batch_file_uploader(batch_use_case: BatchProcessDocumentUseCase):
    """Render the batch file upload interface with progress feedback."""
    source = st.radio(
        "Source",
        ["Upload files", "Local path"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if source == "Upload files":
        _render_upload_tab(batch_use_case)
    else:
        _render_local_path_tab(batch_use_case)


def _render_upload_tab(batch_use_case: BatchProcessDocumentUseCase):
    """Render the drag-and-drop file uploader."""
    ext_list = sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=ext_list,
        accept_multiple_files=True,
        help=f"Supported: {', '.join(e.upper() for e in ext_list)} (max 10MB each)",
    )

    if uploaded_files:
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
                documents = [
                    ProcessDocumentRequest(
                        content=f.getvalue(),
                        filename=f.name,
                        content_type=f.type or "application/octet-stream",
                    )
                    for f in valid_files
                ]
                _process_batch(documents, batch_use_case)


def _render_local_path_tab(batch_use_case: BatchProcessDocumentUseCase):
    """Render the local path input for files/folders."""
    path_input = st.text_input(
        "Path",
        placeholder="/home/user/documents or /home/user/file.pdf",
        help="Enter a file path, folder path, or glob pattern (e.g. /docs/**/*.md)",
    )

    if not path_input:
        return

    resolved_files = _resolve_path(path_input.strip())

    if not resolved_files:
        st.warning("No supported files found at the given path.")
        return

    st.info(f"{len(resolved_files)} file(s) found")
    with st.expander("Files to process"):
        for f in resolved_files:
            st.text(str(f))

    if st.button("Process All Documents", type="primary", use_container_width=True):
        documents = []
        skipped = []
        for file_path in resolved_files:
            if file_path.stat().st_size > settings.max_file_size:
                skipped.append(file_path.name)
                continue
            documents.append(
                ProcessDocumentRequest(
                    content=file_path.read_bytes(),
                    filename=file_path.name,
                    content_type=_get_mime_type(file_path),
                )
            )

        for name in skipped:
            st.warning(f"Skipping {name}: File too large (max 10MB)")

        if documents:
            _process_batch(documents, batch_use_case)
        else:
            st.error("No valid files to process after filtering.")


def _resolve_path(path_str: str) -> list[Path]:
    """Resolve a path string into a list of supported files.

    Supports:
    - Single file path
    - Directory path (recursively finds supported files)
    - Glob patterns (e.g. /docs/**/*.md)
    """
    # Check for glob characters
    if any(c in path_str for c in ("*", "?", "[")):
        base = Path(path_str.split("*")[0].split("?")[0].split("[")[0]).parent
        if not base.exists():
            return []
        pattern = path_str[len(str(base)) :].lstrip("/")
        return sorted(
            f for f in base.glob(pattern) if f.is_file() and f.suffix in SUPPORTED_EXTENSIONS
        )

    path = Path(path_str)
    if not path.exists():
        return []

    if path.is_file():
        return [path] if path.suffix in SUPPORTED_EXTENSIONS else []

    if path.is_dir():
        return sorted(
            f
            for ext in SUPPORTED_EXTENSIONS
            for f in path.rglob(f"*{ext}")
            if f.is_file()
        )

    return []


def _get_mime_type(file_path: Path) -> str:
    """Get MIME type from file extension."""
    mime = EXTENSION_MIME_MAP.get(file_path.suffix.lower())
    if mime:
        return mime
    guessed, _ = mimetypes.guess_type(str(file_path))
    return guessed or "application/octet-stream"


def _process_batch(
    documents: list[ProcessDocumentRequest],
    batch_use_case: BatchProcessDocumentUseCase,
):
    """Process a batch of documents with progress display."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    request = BatchProcessDocumentRequest(documents=documents)

    def update_progress(current: int, total: int, filename: str):
        progress_bar.progress(current / total)
        status_text.text(f"Processing {current}/{total}: {filename}")

    response = batch_use_case.execute(request, progress_callback=update_progress)

    progress_bar.empty()
    status_text.empty()

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

    with st.expander("View Processing Details"):
        for result in response.results:
            if result.response.success:
                st.success(
                    f"**{result.filename}**: {result.response.chunks_created} chunks created"
                )
            else:
                st.error(f"**{result.filename}**: {result.response.message}")
