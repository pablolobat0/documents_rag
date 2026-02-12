import logging
import re

import yaml

from src.domain.value_objects.extraction_result import ExtractionResult
from src.domain.value_objects.page_content import PageContent

logger = logging.getLogger(__name__)


class MarkdownProcessor:
    """Markdown processor implementation. Implements ContentExtractorPort."""

    @property
    def supported_content_types(self) -> list[str]:
        return ["text/markdown", "text/x-markdown"]

    def extract_content(self, file_content: bytes) -> ExtractionResult:
        """
        Extract content from a Markdown file, splitting on H1/H2 headers.
        Parses YAML frontmatter if present and strips it from content.

        Args:
            file_content: Raw file bytes

        Returns:
            ExtractionResult with sections, total count, and frontmatter metadata
        """
        text = file_content.decode("utf-8")
        frontmatter, body = self._parse_frontmatter(text)
        sections = self._split_by_headers(body)

        page_contents = []
        for idx, (title, content) in enumerate(sections, start=1):
            if content.strip():
                page_contents.append(
                    PageContent(
                        content=content,
                        page_number=idx,
                        content_type="text",
                        section_title=title,
                    )
                )

        if not page_contents:
            return ExtractionResult(page_contents=[], total_pages=0)

        return ExtractionResult(
            page_contents=page_contents,
            total_pages=len(page_contents),
            document_metadata=frontmatter,
        )

    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
        """
        Extract YAML frontmatter from text.

        Returns:
            Tuple of (frontmatter dict, remaining body text)
        """
        pattern = r"^---\s*\n(.*?)\n---\s*\n?(.*)"
        match = re.match(pattern, text, re.DOTALL)
        if not match:
            return {}, text

        yaml_block = match.group(1)
        body = match.group(2)
        try:
            parsed = yaml.safe_load(yaml_block)
            if not isinstance(parsed, dict):
                logger.warning("Frontmatter is not a YAML mapping, ignoring")
                return {}, text
            return parsed, body
        except yaml.YAMLError as e:
            logger.warning("Failed to parse YAML frontmatter: %s", e)
            return {}, text

    def _split_by_headers(self, text: str) -> list[tuple[str | None, str]]:
        """
        Split markdown by H1/H2 headers, returning (title, content) tuples.

        Args:
            text: Raw markdown text

        Returns:
            List of (section_title, section_content) tuples
        """
        # Pattern matches H1 (# ) or H2 (## ) at start of line
        header_pattern = r"^(#{1,2})\s+(.+)$"
        lines = text.split("\n")

        sections: list[tuple[str | None, str]] = []
        current_title: str | None = None
        current_content: list[str] = []

        for line in lines:
            match = re.match(header_pattern, line)
            if match:
                # Save previous section if it has content
                if current_content or current_title:
                    sections.append((current_title, "\n".join(current_content)))
                current_title = match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)

        # Add final section
        if current_content or current_title:
            sections.append((current_title, "\n".join(current_content)))

        # Handle case where there are no headers
        if not sections and text.strip():
            sections.append((None, text))

        return sections
