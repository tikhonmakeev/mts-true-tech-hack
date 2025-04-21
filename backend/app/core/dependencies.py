from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasicCredentials, HTTPBasic

from app.repositories.file_repository import FileRepository
from app.services.file_service import FileService
from app.services.exc import AuthException

from app.core.database import SessionLocal


def get_file_service() -> FileService:
    return FileService(FileRepository(SessionLocal))
