from langchain.prompts import PromptTemplate
from models.llm_model import get_llm
from typing import Optional, Callable, Any
from pydantic import PrivateAttr
from langchain.agents import Tool


class SummaryAgentTool(Tool):
    name: str = "summary_agent_tool"
    description: str = "Генерирует краткое резюме диалога для CRM"
    _summary_agent: Any = PrivateAttr()

    def __init__(
        self,
        name: str="summary_agent_tool",
        description: str="Генерирует краткое резюме диалога для CRM",
        func: Optional[Callable] = None,
        **kwargs: Any
    ):
        self._summary_agent = SummaryAgent()
        super().__init__(name=name, func=lambda x: x, description=description, **kwargs)

    def _run(self, query: str, *args: Any, **kwargs: Any) -> str:
        return self._summary_agent.summarize_dialogue(query)

    async def _arun(self, query: str, *args: Any, **kwargs: Any) -> str:
        return self._run(query, *args, **kwargs)


class SummaryAgent:
    def __init__(self):
        self.llm = get_llm()
        self.prompt = PromptTemplate(
            input_variables=["dialogue"],
            template=(
                "Сформируй краткое и информативное резюме следующего диалога между оператором и клиентом:\n\n"
                "{dialogue}\n\n"
                "Структура резюме:\n"
                "- Причина обращения клиента\n"
                "- Действия оператора\n"
                "- Итог разговора\n\n"
                "Пиши по-деловому и лаконично."
            )
        )
        self.chain = self.prompt | self.llm

    def summarize_dialogue(self, dialogue: str) -> str:
        response = self.chain.run(dialogue=dialogue)
        return response.strip()
