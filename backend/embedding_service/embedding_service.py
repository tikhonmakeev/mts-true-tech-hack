import io
import PyPDF2
import faiss
import numpy as np
from typing import List, Dict
from pathlib import Path
import json
import os
import re
import requests
from datetime import datetime
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

INDEX_STORAGE_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_indices")
METADATA_FILE = os.path.join(INDEX_STORAGE_PATH, "metadata.json")
MWS_API_KEY = os.getenv("MWS_API_KEY")
MWS_EMBEDDING_URL = "https://api.gpt.mws.ru/v1/embeddings"
MWS_EMBEDDING_MODEL = "bge-m3"

os.makedirs(INDEX_STORAGE_PATH, exist_ok=True)

SOURCE_FOLDER = os.getenv("BASE_USER_FILES_DIR", "user_files")
os.makedirs(SOURCE_FOLDER, exist_ok=True)


def load_metadata() -> Dict[str, Dict]:
    if os.path.exists(METADATA_FILE):
        logger.info(f"Loading metadata from {METADATA_FILE}")
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    logger.info("No existing metadata found, starting fresh")
    return {}


def save_metadata(data: Dict[str, Dict]):
    logger.info(f"Saving metadata to {METADATA_FILE}")
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


metadata_storage = load_metadata()


def get_embeddings_from_mws(texts: List[str]) -> np.ndarray:
    logger.info(f"Requesting embeddings for {len(texts)} chunks from MWS API")
    headers = {
        "Authorization": f"Bearer {MWS_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MWS_EMBEDDING_MODEL,
        "input": texts
    }
    response = requests.post(MWS_EMBEDDING_URL, headers=headers, json=payload)
    if response.status_code != 200:
        logger.error(f"Failed to get embeddings: {response.status_code}, {response.text}")
        raise RuntimeError(f"Failed to get embeddings: {response.status_code}, {response.text}")

    embeddings = [item["embedding"] for item in response.json()["data"]]
    logger.info(f"Received embeddings shape: ({len(embeddings)}, {len(embeddings[0]) if embeddings else 0})")
    return np.array(embeddings, dtype=np.float32)


def save_embeddings_for_file(filepath: str):
    try:
        logger.info(f"Processing file '{filepath}'")
        file_path = Path(filepath)
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return {"status": "error", "message": "File not found"}

        all_chunks = []
        chunks_path = os.path.join(INDEX_STORAGE_PATH, f"chunks.json")

        if os.path.exists(chunks_path):
            logger.info(f"Loading existing chunks from {chunks_path}")
            with open(chunks_path, 'r', encoding='utf-8') as f:
                all_chunks = json.load(f)

        with open(file_path, 'rb') as f:
            content = f.read()

        file_type = file_path.suffix.lower()
        logger.info(f"Detected file type: {file_type}")
        if file_type == '.txt':
            chunks = process_text_content(content.decode('utf-8'))
        elif file_type == '.pdf':
            chunks = process_pdf_content(content)
        elif file_type == '.json':
            chunks = process_json_content(content)
        else:
            logger.error(f"Unsupported file type: {file_type}")
            return {"status": "error", "message": f"Unsupported file type: {file_type}"}

        all_chunks.extend(chunks)
        if not all_chunks:
            logger.warning("No valid content found in file after processing")
            return {"status": "error", "message": "No valid content found in file"}

        logger.info(f"Total chunks to embed: {len(all_chunks)}")

        # Генерация эмбеддингов через MWS
        embeddings = get_embeddings_from_mws(all_chunks)

        # FAISS
        index_path = os.path.join(INDEX_STORAGE_PATH, "index.faiss")
        dimension = embeddings.shape[1]

        if os.path.exists(index_path):
            logger.info(f"Loading existing FAISS index from {index_path}")
            index = faiss.read_index(index_path)
            index.reset()
            index.add(embeddings)
        else:
            logger.info(f"Creating new FAISS index with dimension {dimension}")
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)

        faiss.write_index(index, index_path)
        logger.info(f"FAISS index saved to {index_path}")

        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False)
            logger.info(f"Chunks saved to {chunks_path}")

        metadata_storage["global"] = {
            "chunks_path": chunks_path,
            "index_path": index_path,
            "chunks_count": len(all_chunks),
            "timestamp": str(datetime.now())
        }
        save_metadata(metadata_storage)

        logger.info(f"Successfully processed file '{filepath}'")
        return {
            "status": "success",
            "chunks_count": len(all_chunks),
            "index_path": index_path,
            "processed_file": str(file_path)
        }

    except Exception as e:
        logger.exception(f"Error processing file '{filepath}': {e}")
        return {"status": "error", "message": str(e)}


def process_text_content(text: str) -> List[str]:
    logger.debug("Processing text content")
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?]) +', text)
    return split_into_chunks(sentences)


def process_pdf_content(pdf_content: bytes) -> List[str]:
    try:
        logger.debug("Processing PDF content")
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text = " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?]) +', text)
        return split_into_chunks(sentences)
    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        return []


def process_json_content(json_content: bytes) -> List[str]:
    try:
        logger.debug("Processing JSON content")
        data = json.loads(json_content.decode('utf-8'))
        text = json.dumps(data, ensure_ascii=False)
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?]) +', text)
        return split_into_chunks(sentences)
    except Exception as e:
        logger.error(f"JSON processing error: {str(e)}")
        return []


def split_into_chunks(sentences: List[str], max_length: int = 1000) -> List[str]:
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += (sentence + " ").strip()
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk)
    logger.debug(f"Split text into {len(chunks)} chunks")
    return chunks


def get_info() -> Dict:
    logger.info("Retrieving global info")
    current_data = load_metadata()
    if "global" not in current_data:
        logger.warning("Global info not found")
        return {"status": "error", "message": "Global info not found"}
    return current_data["global"]


def process_all_existing_files():
    logger.info(f"Starting processing of existing files in '{SOURCE_FOLDER}'")
    results = []
    for root, _, files in os.walk(SOURCE_FOLDER):
        for filename in files:
            file_path = os.path.join(root, filename)
            logger.info(f"Processing file '{filename}'")
            result = save_embeddings_for_file(file_path)
            results.append(result)
    logger.info("Completed processing all existing files")
    return results


if __name__ == "__main__":
    logger.info("Processing existing files...")
    results = process_all_existing_files()
    for result in results:
        logger.info(f"Processed: {result.get('processed_file', 'unknown')}, status: {result.get('status', 'unknown')}")
    logger.info("Ready to process new files.")
