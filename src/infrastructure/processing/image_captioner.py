import base64
import io

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
from src.domain.prompts.image_captioner import ImageCaptionerPrompts


class LangchainImageCaptioner:
    """Langchain-based image captioning implementation."""

    def __init__(
        self,
        llm: BaseChatModel,
        min_width: int = 100,
        min_height: int = 100,
    ):
        self._llm = llm
        self.min_width = min_width
        self.min_height = min_height

    def _should_process_image(self, image_bytes: bytes) -> bool:
        """Check if image meets minimum size requirements."""
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                width, height = img.size
                return width >= self.min_width and height >= self.min_height
        except Exception:
            return False

    def get_image_summary(self, image_bytes: bytes) -> str | None:
        """Returns image summary if image meets size requirements, otherwise returns None."""
        if not self._should_process_image(image_bytes):
            return None

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        return self._llm.invoke(
            [
                SystemMessage(ImageCaptionerPrompts.CAPTION_IMAGE_SYSTEM_PROMPT),
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": ImageCaptionerPrompts.CAPTION_IMAGE_USER_PROMPT,
                        },
                        {
                            "type": "image",
                            "source_type": "base64",
                            "data": image_b64,
                            "mime_type": "image/jpeg",
                        },
                    ],
                ),
            ]
        ).content
