from langchain.prompts import PromptTemplate
from models.llm_model import get_llm
from typing import Optional, Callable, Any
from pydantic import PrivateAttr
from langchain.agents import Tool


class QualityAssuranceAgentTool(Tool):
    name: str = "quality_assurance_tool"
    description: str = "Анализирует диалог между оператором и клиентом на соответствие стандартам качества"
    _quality_assurance_agent: Any = PrivateAttr()

    def __init__(
        self,
        name: str = "quality_assurance_tool",
        func: Optional[Callable] = None,
        description: str = "Анализирует диалог между оператором и клиентом на соответствие стандартам качества",
        **kwargs: Any
    ):
        self._quality_assurance_agent = QualityAssuranceAgent()
        super().__init__(name=name, func=lambda x: x, description=description, **kwargs)

    def _run(self, query: str, *args: Any, **kwargs: Any) -> str:
        return self._quality_assurance_agent.evaluate_dialogue(query)

    async def _arun(self, query: str, *args: Any, **kwargs: Any) -> str:
        return self._run(query, *args, **kwargs)


class QualityAssuranceAgent:
    def __init__(self):
        self.llm = get_llm()
        self.prompt = PromptTemplate(
            input_variables=["dialogue"],
            template=(
                "Проанализируй следующий диалог между оператором контакт-центра и клиентом:\n\n"
                "{dialogue}\n\n"
                "Оцени его по следующим критериям:\n"
                "- Вежливость оператора\n"
                "- Завершенность диалога\n"
                "- Соблюдение стандартов обслуживания\n"
                "- Адекватность ответов\n\n"
                "Если есть нарушения — предложи конкретные рекомендации. Ответ структурируй чётко."
            )
        )
        self.chain = self.prompt | self.llm

    def evaluate_dialogue(self, dialogue: str) -> str:
        response = self.chain.run(dialogue=dialogue)
        return response.strip()
