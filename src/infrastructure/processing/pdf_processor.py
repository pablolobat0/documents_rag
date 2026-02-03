import io
import logging

import pypdf

from src.domain.value_objects.page_content import PageContent
from src.infrastructure.processing.image_captioner import LangchainImageCaptioner

logger = logging.getLogger(__name__)


class PypdfProcessor:
    """PyPDF-based PDF processor implementation. Implements PdfProcessorPort."""

    def __init__(self, image_captioner: LangchainImageCaptioner):
        self.image_captioner = image_captioner

    def extract_content(self, file_content: bytes) -> tuple[list[PageContent], int]:
        """
        Extract text and image descriptions from a PDF.

        Returns:
            Tuple of (list of PageContent with page numbers, number of pages)
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
                page_number = page_num + 1  # Convert to 1-indexed

                try:
                    text = page.extract_text()
                    if text.strip():
                        documents.append(
                            PageContent(
                                content=text,
                                page_number=page_number,
                                content_type="text",
                            )
                        )

                    for image in page.images:
                        try:
                            image_summary = self.image_captioner.get_image_summary(
                                image.data
                            )
                            if image_summary is not None:
                                documents.append(
                                    PageContent(
                                        content=image_summary,
                                        page_number=page_number,
                                        content_type="image_caption",
                                    )
                                )
                        except Exception as e:
                            logger.warning(
                                "Could not process image on page %d: %s", page_number, e
                            )
                except Exception as e:
                    logger.warning("Could not process page %d: %s", page_number, e)
                    continue

            if not documents:
                raise ValueError("No extractable content found in PDF")

            return documents, pdf_pages

        except Exception as e:
            raise ValueError(f"Failed to process PDF: {str(e)}")
