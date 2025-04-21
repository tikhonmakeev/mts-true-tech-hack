import os

class Settings:
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_indices")
    CHUNKS_FILE_PATH = os.getenv("CHUNKS_FILE_PATH", "./faiss_indices/chunks.json")
    EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "https://api.gpt.mws.ru/v1/embeddings")
    MWS_API_KEY = os.getenv("MWS_API_KEY")
    MWS_CHAT_API_URL = os.getenv("MWS_CHAT_API_URL", "https://api.gpt.mws.ru/v1/chat/completions")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instruct")
    INTENT_MODEL_PATH = os.getenv("INTENT_MODEL_PATH", "./intent_local_model")