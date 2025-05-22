import pprint

from chains.combined_chain import LangChainAgent
from config.settings import settings

from dotenv import load_dotenv
load_dotenv()

import logging

logging.basicConfig(
    level=logging.INFO,  # или DEBUG, если хочешь больше деталей
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

pprint.pprint(settings.__dict__)
langchain_agent = LangChainAgent()
pprint.pprint(langchain_agent.run("Мне совсем не нравятся цены на ваш новый тариф!"))