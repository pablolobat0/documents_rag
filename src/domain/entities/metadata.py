import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Metadata(BaseModel):
    pages: int
    document_name: str | None = Field(
        default=None, description="The name of the document file."
    )
    file_path: str | None = Field(
        default=None, description="The path to the document file."
    )
    file_size: int | None = Field(
        default=None, description="The size of the document file in bytes."
    )
    file_type: str | None = Field(
        default=None, description="The MIME type or file extension."
    )
    created_at: datetime.datetime | None = Field(
        default=None, description="When the document was created."
    )
    processed_at: datetime.datetime | None = Field(
        default=None, description="When the document was processed."
    )


class CurriculumVitae(Metadata):
    name: str | None = Field(default=None, description="The name of the person.")
    email: str | None = Field(
        default=None, description="The email address of the person."
    )
    phone_number: str | None = Field(
        default=None, description="The phone number of the person."
    )
    linkedin_profile: str | None = Field(
        default=None, description="The URL of the person's LinkedIn profile."
    )
    skills: list[str] = Field(default_factory=list, description="A list of skills.")
    experience: list[str] = Field(
        default_factory=list, description="A list of work experiences."
    )
    education: list[str] = Field(
        default_factory=list, description="A list of educational qualifications."
    )


class Receipt(Metadata):
    merchant_name: str | None = Field(
        default=None, description="The name of the merchant."
    )
    merchant_address: str | None = Field(
        default=None, description="The address of the merchant."
    )
    transaction_date: str | None = Field(
        default=None, description="The date of the transaction."
    )
    transaction_time: str | None = Field(
        default=None, description="The time of the transaction."
    )
    total_amount: float | None = Field(
        default=None, description="The total amount of the transaction."
    )
    items: list[str] = Field(
        default_factory=list, description="A list of items purchased."
    )


class DocumentClassification(BaseModel):
    document_type: Literal["receipt", "cv", "none"] = Field(
        description="The type of the document (e.g., 'receipt', 'cv', 'none')."
    )
