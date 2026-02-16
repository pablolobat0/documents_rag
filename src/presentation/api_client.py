import httpx

from config.settings import settings


class ApiClient:
    """HTTP client for communicating with the FastAPI backend."""

    def __init__(self, base_url: str = settings.api_url):
        self._base_url = f"{base_url}{settings.api_prefix}"
        self._headers: dict[str, str] = {}
        if settings.api_key:
            self._headers["X-API-Key"] = settings.api_key

    def send_documents(self, files: list[tuple[str, bytes, str]]) -> dict:
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
                f"{self._base_url}/documents/batch",
                files=multipart_files,
                headers=self._headers,
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
                f"{self._base_url}/chat",
                json={"session_id": session_id, "messages": messages},
                headers=self._headers,
            )
            response.raise_for_status()
            return response.json()


api_client = ApiClient()
