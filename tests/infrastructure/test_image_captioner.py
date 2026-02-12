import io
from unittest.mock import MagicMock

import pytest
from PIL import Image

from src.infrastructure.processing.image_captioner import LangchainImageCaptioner


def _make_image_bytes(width: int, height: int, fmt: str = "PNG") -> bytes:
    """Create a minimal in-memory image of the given size and format."""
    img = Image.new("RGB", (width, height), color="red")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def captioner():
    mock_llm = MagicMock()
    return LangchainImageCaptioner(llm=mock_llm, min_width=100, min_height=100)


class TestShouldProcessImage:
    def test_large_enough(self, captioner):
        img = _make_image_bytes(200, 200)
        assert captioner._should_process_image(img) is True

    def test_exactly_minimum(self, captioner):
        img = _make_image_bytes(100, 100)
        assert captioner._should_process_image(img) is True

    def test_too_small_width(self, captioner):
        img = _make_image_bytes(50, 200)
        assert captioner._should_process_image(img) is False

    def test_too_small_height(self, captioner):
        img = _make_image_bytes(200, 50)
        assert captioner._should_process_image(img) is False

    def test_invalid_bytes_returns_false(self, captioner):
        assert captioner._should_process_image(b"not an image") is False


class TestDetectMimeType:
    def test_png_detected(self, captioner):
        img = _make_image_bytes(10, 10, "PNG")
        assert captioner._detect_mime_type(img) == "image/png"

    def test_jpeg_detected(self, captioner):
        img = _make_image_bytes(10, 10, "JPEG")
        assert captioner._detect_mime_type(img) == "image/jpeg"

    def test_invalid_bytes_defaults_to_jpeg(self, captioner):
        assert captioner._detect_mime_type(b"garbage") == "image/jpeg"
