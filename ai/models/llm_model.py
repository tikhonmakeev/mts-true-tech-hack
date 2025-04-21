from langchain.llms import BaseLLM
import requests
import logging

logger = logging.getLogger(__name__)

class LLMModel(BaseLLM):
    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def _call(self, prompt: str) -> str:
        logger.info(f"Generating LLM response for prompt: {prompt[:50]}...")  # first 50 chars for logging

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Ты помощник, который отвечает на вопросы по заданному контексту."},
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"LLM API error: {response.status_code} {response.text}")
        result = response.json()
        return result['choices'][0]['message']['content'].strip()

    def generate_response(self, query: str, context: str) -> str:
        prompt = f"Контекст: {context}\nВопрос: {query}\nОтвет:"
        return self._call(prompt)
