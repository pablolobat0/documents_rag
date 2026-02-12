import httpx

from config.settings import settings


class ApiClient:
    """HTTP client for communicating with the FastAPI backend."""

    def __init__(self, base_url: str = settings.api_url):
        self._base_url = base_url

    def send_documents(
        self, files: list[tuple[str, bytes, str]]
    ) -> dict:
        """Upload documents to the API.

        Args:
            files: List of (filename, content_bytes, content_type) tuples.

        Returns:
            BatchResponseSchema as a dict.
        """
        multipart_files = [
            ("files", (filename, content, content_type))
            for filename, content, content_type in files
        ]
        with httpx.Client(timeout=300.0) as client:
            response = client.post(
                f"{self._base_url}/api/documents/batch",
                files=multipart_files,
            )
            response.raise_for_status()
            return response.json()

    def send_chat(self, session_id: str, messages: list[dict]) -> dict:
        """Send a chat request to the API.

        Args:
            session_id: The session identifier.
            messages: List of {"role": ..., "content": ...} dicts.

        Returns:
            ChatResponseSchema as a dict.
        """
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self._base_url}/api/chat",
                json={"session_id": session_id, "messages": messages},
            )
            response.raise_for_status()
            return response.json()


api_client = ApiClient()
