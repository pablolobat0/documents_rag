from unittest.mock import patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import verify_api_key

_TEST_KEY = "test-key-value"  # pragma: allowlist secret


@pytest.fixture
def app():
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(verify_api_key)])
    async def protected():
        return {"status": "ok"}

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestApiKeyAuth:
    def test_auth_disabled_when_no_key_configured(self, client):
        with patch("src.api.dependencies.settings") as mock_settings:
            mock_settings.api_key = ""
            response = client.get("/protected")
            assert response.status_code == 200

    def test_valid_key_allowed(self, client):
        with patch("src.api.dependencies.settings") as mock_settings:
            mock_settings.api_key = _TEST_KEY
            response = client.get("/protected", headers={"X-API-Key": _TEST_KEY})
            assert response.status_code == 200

    def test_invalid_key_rejected(self, client):
        with patch("src.api.dependencies.settings") as mock_settings:
            mock_settings.api_key = _TEST_KEY
            response = client.get("/protected", headers={"X-API-Key": "wrong-key"})
            assert response.status_code == 401

    def test_missing_key_rejected(self, client):
        with patch("src.api.dependencies.settings") as mock_settings:
            mock_settings.api_key = _TEST_KEY
            response = client.get("/protected")
            assert response.status_code == 401
