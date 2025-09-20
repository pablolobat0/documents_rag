from langchain_core.language_models import BaseChatModel
from app.schemas.metadata import Metadata


class MetadataService():
    def __init__(self, model: BaseChatModel) -> None:
        self.llm = model
        self.cv_keywords = []
        self.bill_keywords = []

    def extract_metadata(self, document: str, pages: int) -> Metadata:
        # TODO: Implement service with regex and if no match LLM, then extract the structured info
        return Metadata(pages=pages)
        
