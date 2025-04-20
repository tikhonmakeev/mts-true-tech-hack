from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import numpy as np
import json
import os
import requests
import uvicorn
import logging
from dotenv import load_dotenv

load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_indices")
os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

MWS_API_URL_CHAT = "https://api.gpt.mws.ru/v1/chat/completions"
MWS_API_URL_EMBEDDING = "https://api.gpt.mws.ru/v1/embeddings"
MWS_API_KEY = os.getenv("MWS_API_KEY")
print("mws_key", MWS_API_KEY)
MWS_CHAT_MODEL = "llama-3.1-8b-instruct"
MWS_EMBEDDING_MODEL = "bge-m3"

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


def load_chunks() -> List[str]:
    path = os.path.join(FAISS_INDEX_PATH, "chunks.json")
    logger.info(f"Loading chunks from {path}")
    if not os.path.exists(path):
        logger.error(f"Chunks file not found at {path}")
        raise HTTPException(status_code=404, detail="Chunks not found")
    with open(path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks

def load_faiss_index():
    index_path = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    logger.info(f"Loading FAISS index from {index_path}")
    if not os.path.exists(index_path):
        logger.error(f"Index file not found at {index_path}")
        raise HTTPException(status_code=404, detail="Index not found")
    index = faiss.read_index(index_path)
    logger.info("FAISS index loaded successfully")
    return index

def get_mws_embedding(text: str) -> List[float]:
    logger.info("Getting embedding from MWS API")
    headers = {
        "Authorization": f"Bearer {MWS_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MWS_EMBEDDING_MODEL,
        "input": text
    }

    response = requests.post(MWS_API_URL_EMBEDDING, headers=headers, json=payload)
    if response.status_code != 200:
        logger.error(f"MWS Embedding API error: {response.status_code}, {response.text}")
        raise RuntimeError(f"MWS Embedding API error: {response.status_code}, {response.text}")

    result = response.json()
    embedding = result["data"][0]["embedding"]
    logger.info("Embedding received successfully")
    return embedding


def generate_llm_response(query: str, context_chunks: List[str]) -> str:
    logger.info("Generating LLM response")
    context = "\n".join(context_chunks)

    logger.info(f"Using context: {context}")

    prompt = f"""Отвечай на русском языке.
Контекст: {context}
Вопрос: {query}
Ответ:"""

    headers = {
        "Authorization": f"Bearer {MWS_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MWS_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": "Ты помощник, который отвечает на вопросы по заданному контексту."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.6
    }

    response = requests.post(MWS_API_URL_CHAT, headers=headers, json=payload)
    if response.status_code != 200:
        logger.error(f"MWS Chat API error: {response.status_code}, {response.text}")
        raise RuntimeError(f"MWS Chat API error: {response.status_code}, {response.text}")

    result = response.json()
    response_text = result['choices'][0]['message']['content'].strip()
    logger.info("LLM response generated successfully")
    return response_text


@app.post("/query/")
async def handle_query(request: QueryRequest):
    logger.info(f"Received query: {request.query}, top_k: {request.top_k}")
    try:
        index = load_faiss_index()
        chunks = load_chunks()

        query_embedding = get_mws_embedding(request.query)
        query_vector = np.array([query_embedding], dtype=np.float32)

        distances, indices = index.search(query_vector, request.top_k)
        relevant_chunks = [chunks[i] for i in indices[0]]
        logger.info(f"Found {len(relevant_chunks)} relevant chunks")

        response = generate_llm_response(request.query, relevant_chunks)
        logger.info(f"Generated response: {response}")

        return {"response": response}

    except HTTPException as e:
        logger.error(f"HTTPException: {e.detail}")
        raise
    except Exception as e:
        logger.exception("An unexpected error occurred")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info("Starting the application")
    logger.info(f"mws_key{MWS_API_KEY}" )
    uvicorn.run(app, host="0.0.0.0", port=3000)
