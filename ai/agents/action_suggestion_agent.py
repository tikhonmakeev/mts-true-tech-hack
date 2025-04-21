from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from models.llm_model import get_llm
from typing import Optional, Callable, Any
from langchain.agents import Tool

class ActionSuggestionAgentTool(Tool):
    name = "action_suggestion_tool"
    description = "Suggests a helpful action for the operator based on user's intent and emotion"

    def __init__(
        self,
        name: str = "action_suggestion_tool",
        func: Optional[Callable] = None,
        description: str = "Suggests a helpful action for the operator based on user's intent and emotion",
        **kwargs: Any
    ):
        self.agent = ActionSuggestionAgent()
        # Проброс фиктивной func, потому что Tool требует func, но мы переопределим _run
        super().__init__(name=name, func=lambda x: x, description=description, **kwargs)

    def _run(self, query: str, *args: Any, **kwargs: Any) -> str:
        """Ожидается, что query — это строка формата intent|emotion"""
        try:
            intent, emotion = query.split("|")
        except ValueError:
            raise ValueError("Invalid input format. Expected 'intent|emotion'")
        return self.agent.suggest_action(intent.strip(), emotion.strip())

    async def _arun(self, query: str, *args: Any, **kwargs: Any) -> str:
        return self._run(query, *args, **kwargs)


class ActionSuggestionAgent:
    def __init__(self):
        self.llm = get_llm()
        self.prompt = PromptTemplate(
            input_variables=["intent", "emotion", "rag_model_context"],
            template=(
                "На основе намерения клиента: '{intent}', его эмоции: '{emotion}' и контекста от RAG-модели{rag_model_context}, "
                "предложи оператору контакт-центра разумное действие. "
                "Формулируй кратко и по-деловому."
            )
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def suggest_action(self, intent: str, emotion: str) -> str:
        response = self.chain.run(intent=intent, emotion=emotion)
        return response.strip()
