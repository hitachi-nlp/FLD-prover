API_KEY = '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'

from langchain.llms import OpenAI
llm = OpenAI(openai_api_key=API_KEY)

text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))
