from langchain.agents import Tool
from models.intent_classifier import IntentClassifier
import logging


class IntentAgentTool(Tool):
    name = "intent_agent_tool"
    description = "A tool for classifying user intent"

    def __init__(self, model_path: str):
        self.intent_agent = IntentAgent(model_path)

    def _run(self, query: str) -> str:
        """Реализация метода классификации намерений для LangChain."""
        return self.intent_agent.classify_intent(query)

    async def _arun(self, query: str) -> str:
        """Асинхронная версия метода для LangChain."""
        return self._run(query)

class IntentAgent:
    def __init__(self, model_dir: str):
        self.intent_classifier = IntentClassifier(model_dir)

    def classify_intent(self, text: str):
        predicted_dict = self.intent_classifier.predict(text)
        intent, confidence = predicted_dict['intent'], predicted_dict['confidence']
        logging.info(f"Confidence {confidence}")
        return intent
