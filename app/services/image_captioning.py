from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
import base64


class ImageCaptioningService:
    def __init__(self, model: str, base_url: str):
        self.llm = ChatOllama(model=model, base_url=base_url)

    def get_image_summary(self, image_bytes: bytes) -> str:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        new_prompt = """You are an expert at describing images for a Retrieval-Augmented Generation (RAG) system. Your task is to generate a detailed, factual, and descriptive summary of the following image. This summary will be stored in a vector database and used to find the image based on user queries.

Please include the following in your description:
-   **Key Objects:** Identify all important objects in the image.
-   **Scene Description:** Describe the overall scene, setting, and context.
-   **Text:** Transcribe any text present in the image.
-   **Attributes:** Mention colors, shapes, and other important visual attributes.

The goal is to create a rich description that will maximize the chances of this image being retrieved for relevant queries. Be objective and stick to what is visually present in the image."""
        message = HumanMessage(
            content=[
                {"type": "text", "text": new_prompt},
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": image_b64,
                    "mime_type": "image/jpeg",
                },
            ],
        )
        return self.llm.invoke([message]).content
