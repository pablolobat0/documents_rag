class DocumentPrompts:
    """Prompts specific to the document information extraction"""

    CLASSIFICATION_SYSTEM_PROMPT = """
        Classify this document as either 'cv' (curriculum vitae/resume), 'receipt', or 'none'.
        """
    EXTRACT_CV_SYSTEM_PROMPT = """
        Extract the following information from this CV/resume text:

        Extract the person's name, email, phone number, LinkedIn profile, skills, work experience, and education.
        """
    EXTRACT_RECEIPT_SYSTEM_PROMPT = """
        Extract the following information from this receipt text:

        Extract the merchant name, address, transaction date/time, total amount, and items purchased.
        """

    @staticmethod
    def format_classification_prompt(document_text: str) -> str:
        """Dynamic prompt construction"""
        return f"""Analyze the following document text and classify it according to your system instructions:
                --- BEGIN DOCUMENT ---
                {document_text}
                --- END DOCUMENT ---
                """

    @staticmethod
    def format_extraction_prompt(document_text: str) -> str:
        """Dynamic prompt construction"""
        return f"""Analyze the following document text and extract the information according to your system instructions:
                --- BEGIN DOCUMENT ---
                {document_text}
                --- END DOCUMENT ---
                """
