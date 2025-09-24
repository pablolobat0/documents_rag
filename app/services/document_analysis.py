import re
from typing import Union
from app.schemas.metadata import Receipt, CurriculumVitae, DocumentClassification

def classify_document(text: str) -> Union[Receipt, CurriculumVitae, DocumentClassification, None]:
    """
    Classifies a document as a receipt, CV, or none, and extracts data if possible.

    Args:
        text: The text content of the document.

    Returns:
        A Pydantic model representing the extracted data or classification, or None if an error occurs.
    """
    if is_receipt(text):
        # Placeholder for LLM-based data extraction for receipts
        # For now, we'll return a placeholder Receipt object
        return Receipt(pages=1, price="10.00") #Dummy data
    elif is_cv(text):
        # Placeholder for LLM-based data extraction for CVs
        # For now, we'll return a placeholder CurriculumVitae object
        return CurriculumVitae(pages=1, name="John Doe") #Dummy data
    else:
        # Placeholder for LLM-based classification
        # For now, we'll return a placeholder DocumentClassification object
        return DocumentClassification(document_type="none")

def is_receipt(text: str) -> bool:
    """
    Checks if a document is a receipt based on keywords.
    """
    receipt_keywords = ["receipt", "total", "cash", "credit", "invoice"]
    text_lower = text.lower()
    return any(re.search(r'\b' + keyword + r'\b', text_lower) for keyword in receipt_keywords)

def is_cv(text: str) -> bool:
    """
    Checks if a document is a CV based on keywords.
    """
    cv_keywords = ["curriculum vitae", "resume", "experience", "education", "skills"]
    text_lower = text.lower()
    return any(re.search(r'\b' + keyword + r'\b', text_lower) for keyword in cv_keywords)
