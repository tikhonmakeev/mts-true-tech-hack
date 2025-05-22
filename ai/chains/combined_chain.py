from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from agents.intent_agent import IntentAgentTool
from agents.knowledge_agent import KnowledgeAgentTool
from agents.emotion_agent import EmotionAgentTool
from agents.action_suggestion_agent import action_suggestion_tool  # <-- без скобок
from agents.summary_agent import SummaryAgentTool
from agents.quality_assurance_agent import QualityAssuranceAgentTool
from config.settings import settings


class LangChainAgent:
    def __init__(self):
        self.intent_tool = IntentAgentTool(model_path=settings.INTENT_MODEL_PATH)
        self.knowledge_tool = KnowledgeAgentTool(
            faiss_index_path=settings.FAISS_INDEX_PATH,
            chunk_file_path=settings.CHUNKS_FILE_PATH,
            embedding_api_url=settings.EMBEDDING_API_URL,
            api_key=settings.MWS_API_KEY,
            embedding_model=settings.EMBEDDING_MODEL
        )
        self.emotion_tool = EmotionAgentTool()
        self.action_suggestion_tool = action_suggestion_tool  # <-- уже созданный tool
        self.summary_tool = SummaryAgentTool()
        self.quality_tool = QualityAssuranceAgentTool()

    def run(self, input_text: str):
        passthrough = RunnablePassthrough()

        # Параллельный блок — намерение, эмоция, RAG
        parallel_block = RunnableParallel(
            intent=self.intent_tool,
            emotion=self.emotion_tool,
            rag_context=self.knowledge_tool
        )

        with_context = passthrough | RunnableLambda(
            lambda x: {
                **x,
                **parallel_block.invoke(x)
            }
        )

        # Подключаем action suggestion
        action_suggestion = with_context | RunnableLambda(
            lambda x: {
                **x,
                "advice": self.action_suggestion_tool.invoke({
                    "intent": x["intent"],
                    "emotion": x["emotion"],
                    "rag_context": x["rag_context"]
                })
            }
        )

        # Саммари
        summary = action_suggestion | RunnableLambda(
            lambda x: {
                **x,
                "summary": self.summary_tool.invoke(x)
            }
        )

        # Оценка качества
        quality = summary | RunnableLambda(
            lambda x: {
                **x,
                "quality": self.quality_tool.invoke(x)
            }
        )

        final_chain = quality

        result = final_chain.invoke({"input": input_text})
        return result
