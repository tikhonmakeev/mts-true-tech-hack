from typing import Optional, Callable, Any

from langchain.agents import Tool
from models.emotion_model import EmotionModel
import logging

class EmotionAgentTool(Tool):
    name = "emotion_agent_tool"
    description = "A tool for detecting emotions in Russian text"

    def __init__(self, name: str, func: Optional[Callable], description: str, **kwargs: Any):
        super().__init__(name, func, description, **kwargs)
        self.emotion_agent = EmotionAgent()

    def _run(self, query: str, *args: Any, **kwargs: Any) -> str:
        return self.emotion_agent.classify_emotion(query)

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
