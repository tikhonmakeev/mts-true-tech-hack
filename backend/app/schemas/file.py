from datetime import datetime

from pydantic import BaseModel


class File(BaseModel):
    id: str | None = None
    filename: str
    created_at: datetime | None = None


class FileRequest(BaseModel):
    filename: str