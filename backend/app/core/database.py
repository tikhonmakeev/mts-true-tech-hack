from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
import os

DATABASE_URL = os.getenv(
    "DB_URL", "postgresql+asyncpg://user:password@db:5432/dbname")

engine = create_async_engine(DATABASE_URL)
SessionLocal = async_sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession)
