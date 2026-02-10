import re

from src.domain.value_objects.page_content import PageContent


class MarkdownProcessor:
    """Markdown processor implementation. Implements ContentExtractorPort."""

    @property
    def supported_content_types(self) -> list[str]:
        return ["text/markdown", "text/x-markdown"]

    def extract_content(self, file_content: bytes) -> tuple[list[PageContent], int]:
        """
        Extract content from a Markdown file, splitting on H1/H2 headers.

        Args:
            file_content: Raw file bytes

        Returns:
            Tuple of (list of PageContent with section numbers, total sections)
        """
        text = file_content.decode("utf-8")
        sections = self._split_by_headers(text)

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

        # Handle edge case: empty file
        if not page_contents:
            return [], 0

        return page_contents, len(page_contents)

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
