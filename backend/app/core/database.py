from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
import os

DATABASE_URL = os.getenv(
    "DB_URL", "postgresql+asyncpg://user:password@db:5432/dbname")

engine = create_async_engine(DATABASE_URL)
SessionLocal = sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession)
Base = declarative_base()
