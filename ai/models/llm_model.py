from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
import logging
import os


from pydantic import Field

from config.settings import settings

logger = logging.getLogger(__name__)
MAX_TOKENS=120


class LLMModel(LLM):
    api_url: str = Field(...)
    api_key: str = Field(...)
    model: str = Field(...)

    @property
    def _llm_type(self) -> str:
        return "custom_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        logger.info(f"Generating LLM response for prompt: {prompt[:50]}...")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "max_tokens": MAX_TOKENS,
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Ты помощник, который отвечает на вопросы по заданному контексту."},
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"LLM API error: {response.status_code} {response.text}")
        content = response.json()['choices'][0]['message']['content'].strip()

        if stop:
            for stop_word in stop:
                if stop_word in content:
                    content = content.split(stop_word)[0]
        return content

    def generate_response(self, query: str, context: str) -> str:
        prompt = f"Контекст: {context}\nВопрос: {query}\nОтвет:"
        return self._call(prompt)

def get_llm() -> LLMModel:
    api_url = settings.MWS_CHAT_API_URL
    api_key = settings.MWS_API_KEY
    model = settings.LLM_MODEL

    return LLMModel(api_url=api_url, api_key=api_key, model=model)
