from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    FAISS_INDEX_PATH: str = "faiss_indices/index.faiss"
    CHUNKS_FILE_PATH: str = "faiss_indices/chunks.json"
    EMBEDDING_API_URL: str = "https://api.gpt.mws.ru/v1/embeddings"
    MWS_API_KEY: str
    MWS_CHAT_API_URL: str = "https://api.gpt.mws.ru/v1/chat/completions"
    EMBEDDING_MODEL: str = "bge-m3"
    LLM_MODEL: str = "llama-3.1-8b-instruct"
    INTENT_MODEL_PATH: str = "intent_local_model"

    class Config:
        env_file = "../.env"

settings = Settings()
