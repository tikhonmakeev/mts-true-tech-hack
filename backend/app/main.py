from fastapi import FastAPI
from app.core.database import engine
from app.models import Base

app = FastAPI(title="My FastAPI Project", version="0.1.0")


# Тут можно доделать миграции
@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.get("/")
async def root():
    return {"message": "Hello World"}
