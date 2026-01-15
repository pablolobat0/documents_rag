from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Metadata:
    """Base metadata for documents."""

    pages: int
    document_name: str | None = None
    file_path: str | None = None
    file_size: int | None = None
    file_type: str | None = None
    created_at: datetime | None = None
    processed_at: datetime | None = None


@dataclass
class CurriculumVitae(Metadata):
    """Metadata for CV/Resume documents."""

    name: str | None = None
    email: str | None = None
    phone_number: str | None = None
    linkedin_profile: str | None = None
    skills: list[str] = field(default_factory=list)
    experience: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)


@dataclass
class Receipt(Metadata):
    """Metadata for receipt documents."""

    merchant_name: str | None = None
    merchant_address: str | None = None
    transaction_date: str | None = None
    transaction_time: str | None = None
    total_amount: float | None = None
    items: list[str] = field(default_factory=list)
