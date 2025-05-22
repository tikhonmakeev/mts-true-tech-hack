from typing import Optional, Callable, Any

from langchain.agents import Tool
from pydantic import PrivateAttr

from models.intent_classifier import IntentClassifier
import logging


class IntentAgentTool(Tool):
    name: str = "intent_agent_tool"
    description: str = "A tool for classifying user intent"
    _intent_agent: Any = PrivateAttr()

    def __init__(self, model_path: str, name: str = "intent_agent_tool",
                 description: str = "A tool for classifying user intent",
                 func: Optional[Callable] = None, **kwargs: Any):
        super().__init__(name=name, func=lambda x: x, description=description, **kwargs)
        self._intent_agent = IntentAgent(model_path)

    def _run(self, query: str, *args: Any, **kwargs: Any) -> str:
        return self._intent_agent.classify_intent(query)

    async def _arun(self, query: str, *args: Any, **kwargs: Any) -> str:
        return self._run(query)

class IntentAgent:
    def __init__(self, model_dir: str):
        self.intent_classifier = IntentClassifier(model_dir)

    def classify_intent(self, text: str):
        predicted_dict = self.intent_classifier.predict(text)
        intent, confidence = predicted_dict['intent'], predicted_dict['confidence']
        logging.info(f"Confidence {confidence}")
        return intent
