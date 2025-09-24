import json
from datetime import datetime
from pathlib import Path
from typing import Union
from app.schemas.metadata import Metadata, CurriculumVitae, Receipt


class MetadataStorageService:
    def __init__(self, storage_dir: str = "document_metadata"):
        self.storage_dir = Path(storage_dir)
        # Ensure the directory exists with parents if needed
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def generate_filename(
        self, document_name: str | None, file_type: str | None
    ) -> str:
        """
        Generate a unique filename for the metadata JSON file.
        """
        if document_name:
            base_name = Path(document_name).stem
        else:
            base_name = f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if file_type and not file_type.startswith("."):
            base_name = f"{base_name}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{base_name}_{timestamp}.json"

    def save_metadata(self, metadata: Union[Metadata, CurriculumVitae, Receipt]) -> str:
        """
        Save metadata to a JSON file and return the file path.
        """
        # Ensure directory exists before saving
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Convert metadata to dict
        metadata_dict = metadata.model_dump()

        # Handle datetime serialization
        if metadata_dict.get("created_at"):
            metadata_dict["created_at"] = metadata_dict["created_at"].isoformat()
        if metadata_dict.get("processed_at"):
            metadata_dict["processed_at"] = metadata_dict["processed_at"].isoformat()

        # Add document type field
        if isinstance(metadata, CurriculumVitae):
            metadata_dict["document_type"] = "cv"
        elif isinstance(metadata, Receipt):
            metadata_dict["document_type"] = "receipt"
        else:
            metadata_dict["document_type"] = "unknown"

        # Generate filename
        filename = self.generate_filename(
            metadata_dict.get("document_name"), metadata_dict.get("file_type")
        )
        file_path = self.storage_dir / filename

        metadata_dict["file_path"] = str(file_path)

        # Save to JSON file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

        return str(file_path)

    def load_metadata(
        self, filename: str
    ) -> Union[Metadata, CurriculumVitae, Receipt, None]:
        """
        Load metadata from a JSON file.
        """
        file_path = self.storage_dir / filename
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Convert string timestamps back to datetime
            if data.get("created_at"):
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            if data.get("processed_at"):
                data["processed_at"] = datetime.fromisoformat(data["processed_at"])

            # Determine the document type and create appropriate object
            document_type = data.get("document_type", "unknown")

            if document_type == "cv":
                return CurriculumVitae(**data)
            elif document_type == "receipt":
                return Receipt(**data)
            else:
                return Metadata(**data)

        except Exception:
            return None

    def list_metadata_files(self) -> list[str]:
        """
        List all metadata JSON files in the storage directory.
        """
        return [f.name for f in self.storage_dir.glob("*.json")]

    def delete_metadata(self, filename: str) -> bool:
        """
        Delete a metadata JSON file.
        """
        file_path = self.storage_dir / filename
        try:
            file_path.unlink()
            return True
        except FileNotFoundError:
            return False
