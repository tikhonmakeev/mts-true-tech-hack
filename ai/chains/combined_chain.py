from langchain.chains import SequentialChain
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from agents.intent_agent import IntentAgent, IntentAgentTool
from agents.knowledge_agent import KnowledgeAgentTool
from agents.action_suggestion_agent import ActionSuggestionAgent
from agents.summary_agent import SummaryAgent
from agents.quality_assurance_agent import QualityAssuranceAgent
from models.llm_model import LLMModel

from config.settings import Settings


class LangChainAgent:
    def __init__(self, intent_model_dir: str, faiss_index_path: str, chunk_file_path: str, embedding_api_url: str, api_key: str, embedding_model: str):
        self.intent_agent = IntentAgent(intent_model_dir)
        self.knowledge_agent_tool = KnowledgeAgentTool(faiss_index_path, chunk_file_path, embedding_api_url, api_key, embedding_model)
        self.action_suggestion_agent = ActionSuggestionAgent()
        self.summary_agent = SummaryAgent()
        self.quality_assurance_agent = QualityAssuranceAgent()

        self.memory = ConversationBufferMemory()

    @staticmethod
    def create_agent_chain():
        tools = [
            KnowledgeAgentTool(faiss_index_path=Settings.FAISS_INDEX_PATH, chunk_file_path=Settings.CHUNKS_FILE_PATH,
                               embedding_api_url=Settings.EMBEDDING_API_URL, api_key=Settings.MWS_API_KEY,
                               embedding_model=Settings.EMBEDDING_MODEL),
            IntentAgentTool(model_path=Settings.INTENT_MODEL_PATH),
            # Тут остальные типа для Action Suggestion, Summary и Quality Assurance
        ]

        llm = LLMModel(Settings.MWS_CHAT_API_URL, Settings.MWS_API_KEY, Settings.LLM_MODEL)  # или другой LLM по вашему выбору

        # Инициализация агента с использованием инструментов и LLM
        agent = initialize_agent(
            tools,
            llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        return agent

    def run(self, input_text: str):
        chain = self.create_agent_chain()
        result = chain.run(input_text)
        return result
