import base64
import io

from langchain_core.messages import HumanMessage
from PIL import Image

from src.infrastructure.llm.ollama import OllamaLLM


class OllamaImageCaptioner:
    """Ollama-based image captioning implementation."""

    def __init__(
        self,
        ollama: OllamaLLM,
        min_width: int = 100,
        min_height: int = 100,
    ):
        self._llm = ollama.get_image_captioning_model()
        self.min_width = min_width
        self.min_height = min_height

    def should_process_image(self, image_bytes: bytes) -> bool:
        """Check if image meets minimum size requirements."""
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                width, height = img.size
                return width >= self.min_width and height >= self.min_height
        except Exception:
            return False

    def get_image_summary(self, image_bytes: bytes) -> str | None:
        """Returns image summary if image meets size requirements, otherwise returns None."""
        if not self.should_process_image(image_bytes):
            return None

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt = """You are an expert at describing images for a Retrieval-Augmented Generation (RAG) system. Your task is to generate a detailed, factual, and descriptive summary of the following image. This summary will be stored in a vector database and used to find the image based on user queries.

Please include the following in your description:
-   **Key Objects:** Identify all important objects in the image.
-   **Scene Description:** Describe the overall scene, setting, and context.
-   **Text:** Transcribe any text present in the image.
-   **Attributes:** Mention colors, shapes, and other important visual attributes.

The goal is to create a rich description that will maximize the chances of this image being retrieved for relevant queries. Be objective and stick to what is visually present in the image."""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": image_b64,
                    "mime_type": "image/jpeg",
                },
            ],
        )
        return self._llm.invoke([message]).content
