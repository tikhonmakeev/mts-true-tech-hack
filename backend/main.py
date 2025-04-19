from fastapi import FastAPI
from app.core.database import engine
from app.models import Base

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
