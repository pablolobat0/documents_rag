from pydantic import BaseModel


class Metadata(BaseModel):
    pages: int

class CurriculumVitae(Metadata):
    name: str

class Receipt(Metadata):
    price: str
