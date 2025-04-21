from models.llm_model import LLMModel

model = LLMModel(api_url='https://api.gpt.mws.ru/v1/chat/completions', api_key='sk-KNo006G2a48UVE3IxFlQEQ', model='llama-3.1-8b-instruct')
context = "Искусственный интеллект - это область информатики, которая занимается созданием систем, способных выполнять задачи, требующие человеческого интеллекта."
query = "Что такое искусственный интеллект?"

response = model.generate_response(query, context)
print(response)