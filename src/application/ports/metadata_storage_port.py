from typing import Protocol, Union

from src.domain.entities.metadata import CurriculumVitae, Metadata, Receipt


class MetadataStoragePort(Protocol):
    """Port for metadata storage operations."""

    def save_metadata(
        self, metadata: Union[Metadata, CurriculumVitae, Receipt]
    ) -> str:
        """Save metadata and return the file path."""
        ...

    def load_metadata(
        self, filename: str
    ) -> Union[Metadata, CurriculumVitae, Receipt, None]:
        """Load metadata from storage."""
        ...

    def list_metadata_files(self) -> list[str]:
        """List all metadata files."""
        ...

    def delete_metadata(self, filename: str) -> bool:
        """Delete a metadata file."""
        ...
