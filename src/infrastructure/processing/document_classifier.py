import re

from src.domain.entities.classification import DocumentClassification
from src.domain.ports.document_classifier_port import DocumentType
from src.domain.ports.llm_port import LLMPort
from src.domain.prompts.document_processing import DocumentPrompts

CV_THRESHOLD = 8.0
RECEIPT_THRESHOLD = 8.0

CV_KEYWORDS = {
    "curriculum vitae": 5,
    "resume": 5,
    "cv": 4,
    "experience": 3,
    "education": 3,
    "skills": 3,
    "work history": 3,
    "employment": 3,
    "qualifications": 2,
    "professional summary": 4,
    "career objective": 2,
    "job title": 2,
    "company": 1,
    "position": 2,
    "duration": 1,
    "degree": 3,
    "university": 3,
    "certification": 2,
    "languages": 1,
    "portfolio": 2,
    "projects": 1,
    "references": 2,
    "contact information": 2,
    "phone": 1,
    "email": 1,
    "linkedin": 2,
    "github": 1,
    "location": 1,
    "profile": 1,
    "career": 1,
}

RECEIPT_KEYWORDS = {
    "receipt": 5,
    "invoice": 4,
    "bill": 3,
    "payment": 3,
    "transaction": 2,
    "total": 4,
    "subtotal": 3,
    "tax": 2,
    "amount due": 3,
    "amount paid": 3,
    "cash": 3,
    "credit": 3,
    "debit": 3,
    "purchase": 2,
    "sale": 2,
    "order": 2,
    "customer": 1,
    "merchant": 2,
    "store": 2,
    "shop": 2,
    "restaurant": 2,
    "date": 1,
    "time": 1,
    "item": 2,
    "quantity": 2,
    "price": 3,
    "cost": 2,
    "discount": 2,
    "change": 2,
    "balance": 1,
    "paid": 2,
    "due": 1,
    "transaction id": 2,
    "order number": 2,
    "invoice number": 2,
    "payment method": 2,
    "card": 2,
    "cash register": 3,
    "pos": 3,
    "store number": 1,
    "location": 1,
    "address": 1,
    "phone": 1,
    "thank you": 2,
    "have a nice day": 1,
    "keep the receipt": 2,
}


class KeywordClassifier:
    """Document classifier with keyword scoring and LLM fallback."""

    def __init__(self, llm: LLMPort):
        self._llm = llm

    def _calculate_cv_score(self, text: str) -> float:
        """Calculate CV score based on weighted keyword matches."""
        text_lower = text.lower()
        total_score = 0.0

        for keyword, weight in CV_KEYWORDS.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
                total_score += weight

        return total_score

    def _calculate_receipt_score(self, text: str) -> float:
        """Calculate receipt score based on weighted keyword matches."""
        text_lower = text.lower()
        total_score = 0.0

        for keyword, weight in RECEIPT_KEYWORDS.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
                total_score += weight

        return total_score

    def classify(self, text: str) -> DocumentType:
        """Classify a document using keyword scoring with LLM fallback."""
        cv_score = self._calculate_cv_score(text)
        receipt_score = self._calculate_receipt_score(text)

        if cv_score >= CV_THRESHOLD and cv_score > receipt_score:
            return DocumentType.CV
        elif receipt_score >= RECEIPT_THRESHOLD and receipt_score > cv_score:
            return DocumentType.RECEIPT
        else:
            return self._classify_with_llm(text)

    def _classify_with_llm(self, text: str) -> DocumentType:
        """Use LLM to classify document when keyword scoring is inconclusive."""
        try:
            result = self._llm.with_structured_output(DocumentClassification).invoke(
                [
                    ("system", DocumentPrompts.CLASSIFICATION_SYSTEM_PROMPT),
                    ("human", DocumentPrompts.format_classification_prompt(text)),
                ]
            )

            if result.document_type == "cv":
                return DocumentType.CV
            elif result.document_type == "receipt":
                return DocumentType.RECEIPT
            else:
                return DocumentType.UNKNOWN
        except Exception:
            return DocumentType.UNKNOWN
