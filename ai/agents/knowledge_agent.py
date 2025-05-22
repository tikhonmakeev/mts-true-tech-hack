import faiss
import json
import numpy as np
import requests
import logging
from langchain.agents import Tool
from typing import List, Any, Optional, Callable
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)

class KnowledgeAgentTool(Tool):
    name: str = "knowledge_agent_tool"
    description: str = "A tool for searching relevant chunks from the knowledge base"
    _knowledge_agent: Any = PrivateAttr()

    def __init__(self, faiss_index_path: str, chunk_file_path: str, embedding_api_url: str, api_key: str,
                 embedding_model: str, name: str = "knowledge_agent_tool",
                 description: str="A tool for searching relevant chunks from the knowledge base",
                 func: Optional[Callable] = None, **kwargs: Any):
        super().__init__(name=name, func=lambda x: x, description=description, **kwargs)
        self._knowledge_agent = KnowledgeAgent(faiss_index_path, chunk_file_path, embedding_api_url, api_key, embedding_model)

    def _run(self, query: str, *args: Any, **kwargs: Any) -> str:
        relevant_chunks = self._knowledge_agent.search_chunks(query, top_k=3)
        return "\n".join(relevant_chunks)

    async def _arun(self, query: str, *args: Any, **kwargs: Any) -> str:
        return self._run(query)


class KnowledgeAgent:
    def __init__(self, faiss_index_path: str, chunk_file_path: str, embedding_api_url: str, api_key: str, embedding_model: str):
        self.faiss_index_path = faiss_index_path
        self.chunk_file_path = chunk_file_path
        self.embedding_api_url = embedding_api_url
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.index = self.load_faiss_index()
        self.chunks = self.load_chunks()

    def load_chunks(self) -> List[str]:
        logger.info(f"Loading chunks from {self.chunk_file_path}")
        with open(self.chunk_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_faiss_index(self):
        logger.info(f"Loading FAISS index from {self.faiss_index_path}")
        return faiss.read_index(self.faiss_index_path)

    def get_embedding_from_api(self, text: str) -> np.ndarray:
        logger.info(f"Getting embedding for text: {text[:50]}...")  # first 50 chars for logging
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.embedding_model,
            "input": text
        }
        response = requests.post(self.embedding_api_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Embedding API error: {response.status_code} {response.text}")
        result = response.json()
        return np.array(result["data"][0]["embedding"], dtype=np.float32)

    def search_chunks(self, query: str, top_k: int = 3) -> List[str]:
        query_embedding = self.get_embedding_from_api(query)
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), top_k)
        relevant_chunks = [self.chunks[i] for i in indices[0]]
        return relevant_chunks
