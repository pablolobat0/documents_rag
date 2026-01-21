class DocumentPrompts:
    """Prompts specific to document information extraction."""

    CLASSIFICATION_SYSTEM_PROMPT = """Classify this document as either 'cv' (curriculum vitae/resume), 'receipt', or 'none'."""

    EXTRACT_CV_SYSTEM_PROMPT = """Extract the following information from this CV/resume text:

Extract the person's name, email, phone number, LinkedIn profile, skills, work experience, and education."""

    EXTRACT_RECEIPT_SYSTEM_PROMPT = """Extract the following information from this receipt text:

Extract the merchant name, address, transaction date/time, total amount, and items purchased."""

    DOCUMENT_USER_PROMPT = """Analyze the following document text and {task} according to your system instructions:

--- BEGIN DOCUMENT ---
{document_text}
--- END DOCUMENT ---"""

    @staticmethod
    def format_classification_prompt(document_text: str) -> str:
        """Format the classification user prompt with document text."""
        return DocumentPrompts.DOCUMENT_USER_PROMPT.format(
            task="classify it", document_text=document_text
        )

    @staticmethod
    def format_extraction_prompt(document_text: str) -> str:
        """Format the extraction user prompt with document text."""
        return DocumentPrompts.DOCUMENT_USER_PROMPT.format(
            task="extract the information", document_text=document_text
        )
