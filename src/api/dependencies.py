from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from config.settings import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(_api_key_header),
) -> None:
    if not settings.api_key:
        return
    if not api_key or api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
