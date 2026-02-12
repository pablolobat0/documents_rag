import mimetypes
from pathlib import Path

import streamlit as st

from config.settings import settings
from src.presentation.api_client import api_client
from src.presentation.state.session_state import increment_document_count

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".md"}
EXTENSION_MIME_MAP = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".txt": "text/plain",
}


def render_batch_file_uploader():
    """Render the batch file upload interface with progress feedback."""
    source = st.radio(
        "Source",
        ["Upload files", "Local path"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if source == "Upload files":
        _render_upload_tab()
    else:
        _render_local_path_tab()


def _render_upload_tab():
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
                files_to_send = [
                    (
                        f.name,
                        f.getvalue(),
                        f.type or "application/octet-stream",
                    )
                    for f in valid_files
                ]
                _process_batch(files_to_send)


def _render_local_path_tab():
    """Render the local path input for files/folders."""
    path_input = st.text_input(
        "Path",
        placeholder="/home/user/documents or /home/user/file.pdf",
        help="Enter a file path, folder path, or glob pattern (e.g. /docs/**/*.md)",
    )

    # Scan button — file resolution only happens on click
    if st.button("Scan", use_container_width=True):
        if not path_input:
            st.warning("Enter a path first.")
            return
        resolved = _resolve_path(path_input.strip())
        st.session_state["_scanned_files"] = resolved

    resolved_files: list[Path] = st.session_state.get("_scanned_files", [])

    if not resolved_files:
        return

    st.info(f"{len(resolved_files)} file(s) found")
    with st.expander("Files to process"):
        for f in resolved_files:
            st.text(str(f))

    if st.button("Process All Documents", type="primary", use_container_width=True):
        files_to_send = []
        skipped = []
        for file_path in resolved_files:
            if file_path.stat().st_size > settings.max_file_size:
                skipped.append(file_path.name)
                continue
            files_to_send.append(
                (
                    file_path.name,
                    file_path.read_bytes(),
                    _get_mime_type(file_path),
                )
            )

        for name in skipped:
            st.warning(f"Skipping {name}: File too large (max 10MB)")

        if files_to_send:
            _process_batch(files_to_send)
        else:
            st.error("No valid files to process after filtering.")


def _resolve_path(path_str: str) -> list[Path]:
    """Resolve a path string into a list of supported files."""
    if any(c in path_str for c in ("*", "?", "[")):
        base = Path(path_str.split("*")[0].split("?")[0].split("[")[0]).parent
        if not base.exists():
            return []
        pattern = path_str[len(str(base)) :].lstrip("/")
        return sorted(
            f
            for f in base.glob(pattern)
            if f.is_file() and f.suffix in SUPPORTED_EXTENSIONS
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


def _process_batch(files: list[tuple[str, bytes, str]]):
    """Process a batch of documents via the API."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Uploading {len(files)} file(s) to API...")
    progress_bar.progress(0.5)

    try:
        result = api_client.send_documents(files)
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"API error: {e}")
        return

    progress_bar.empty()
    status_text.empty()

    _display_batch_results(result)


def _display_batch_results(response: dict):
    """Display batch processing results."""
    total = response["total_documents"]
    successful = response["successful"]
    failed = response["failed"]
    results = response["results"]

    total_chunks = sum(r["chunks_created"] for r in results if r["success"])

    if failed == 0 and total > 0:
        st.success(
            f"All {total} documents processed successfully! "
            f"Total chunks created: {total_chunks}"
        )
    else:
        st.warning(
            f"Processed {total} documents: "
            f"{successful} succeeded, {failed} failed"
        )

    for _ in range(successful):
        increment_document_count()

    with st.expander("View Processing Details"):
        for r in results:
            if r["success"]:
                st.success(
                    f"**{r['filename']}**: {r['chunks_created']} chunks created"
                )
            else:
                st.error(f"**{r['filename']}**: {r['message']}")
