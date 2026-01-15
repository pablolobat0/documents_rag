import io

import pypdf

from src.infrastructure.processing.image_captioner import OllamaImageCaptioner


class PypdfProcessor:
    """PyPDF-based PDF processor implementation. Implements PdfProcessorPort."""

    def __init__(self, image_captioner: OllamaImageCaptioner):
        self.image_captioner = image_captioner

    def extract_content(self, file_content: bytes) -> tuple[list[str], int]:
        """
        Extract text and image descriptions from a PDF.

        Returns:
            Tuple of (list of text/image content, number of pages)
        """
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = pypdf.PdfReader(pdf_file)

            if pdf_reader.is_encrypted:
                raise ValueError("Encrypted PDF files are not supported")

            pdf_pages = len(pdf_reader.pages)

            if pdf_pages == 0:
                raise ValueError("PDF has no pages")

            documents = []

            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text.strip():
                        documents.append(text)

                    for image in page.images:
                        try:
                            image_summary = self.image_captioner.get_image_summary(
                                image.data
                            )
                            if image_summary is not None:
                                documents.append(image_summary)
                        except Exception as e:
                            print(
                                f"Warning: Could not process image on page {page_num}: {e}"
                            )
                except Exception as e:
                    print(f"Warning: Could not process page {page_num}: {e}")
                    continue

            if not documents:
                raise ValueError("No extractable content found in PDF")

            return documents, pdf_pages

        except Exception as e:
            raise ValueError(f"Failed to process PDF: {str(e)}")
