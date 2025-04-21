from app.repositories.file_repository import FileRepository
from app.schemas.file import File
from fastapi import UploadFile
import os

from app.core.config import BASE_USER_FILES_DIR
from embedding_service import embedding_service

class FileService:
    def __init__(self, file_repository: FileRepository):
        self.file_repository = file_repository

    async def create_file(self, file: UploadFile) -> File:
        file = await self.file_repository.create_file(BASE_USER_FILES_DIR, file.filename, file.file)
        filepath = os.path.join(BASE_USER_FILES_DIR, file.filename)
        embedding_service.save_embeddings_for_file(filepath)
        return file

    async def get_all_files(self) -> list[File]:
        return await self.file_repository.get_all_files()

    @staticmethod
    def check_file_exists_locally(filename: str):
        return FileRepository.check_file_exists_locally(BASE_USER_FILES_DIR, filename)