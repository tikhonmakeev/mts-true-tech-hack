from typing import Optional, Callable, Any
from pydantic import PrivateAttr

from langchain.agents import Tool
from models.emotion_model import EmotionModel
import logging

class EmotionAgentTool(Tool):
    name: str = "emotion_agent_tool"
    description: str = "A tool for detecting emotions in Russian text"
    _emotion_agent: Any = PrivateAttr()

    def __init__(self, name: str = "emotion_agent_tool",
                 func: Optional[Callable] = None,
                 description: str = "A tool for detecting emotions in Russian text",
                 **kwargs: Any):
        super().__init__(name=name, func=lambda x: x, description=description, **kwargs)
        self._emotion_agent = EmotionAgent()

    def _run(self, query: str, *args: Any, **kwargs: Any) -> str:
        return self._emotion_agent.classify_emotion(query)

    async def _arun(self, query: str, *args: Any, **kwargs: Any) -> str:
        return self._run(query)

class EmotionAgent:
    def __init__(self):
        self.emotion_model = EmotionModel()

    def classify_emotion(self, text: str) -> str:
        result = self.emotion_model.predict_emotion(text)
        emotion, confidence = result["emotion"], result["confidence"]
        logging.info(f"Detected emotion: {emotion} with confidence {confidence}")
        return f"{emotion} ({confidence:.2f})"
