from langchain.prompts import PromptTemplate
from models.llm_model import get_llm
from langchain_core.tools import StructuredTool


class ActionSuggestionAgent:
    def __init__(self):
        self.llm = get_llm()
        self.prompt = PromptTemplate(
            input_variables=["intent", "emotion", "rag_model_context"],
            template=(
                "На основе намерения клиента: '{intent}', его эмоции: '{emotion}' и контекста от RAG-модели: '{rag_model_context}', "
                "предложи оператору контакт-центра разумное действие. "
                "Формулируй кратко и по-деловому."
            )
        )
        # теперь используем пайплайн: prompt → llm
        self.chain = self.prompt | self.llm

    def suggest_action(self, intent: str, emotion: str, rag_model_context: str) -> str:
        response = self.chain.invoke({
            "intent": intent,
            "emotion": emotion,
            "rag_model_context": rag_model_context
        })
        return response.strip() if isinstance(response, str) else str(response)


agent = ActionSuggestionAgent()

def suggest_action_tool_func(intent: str, emotion: str, rag_context: str) -> str:
    return agent.suggest_action(intent, emotion, rag_context)

action_suggestion_tool = StructuredTool.from_function(
    func=suggest_action_tool_func,
    name="action_suggestion_tool",
    description="Предлагает оператору разумное действие по намерению, эмоции и знанию.",
)
