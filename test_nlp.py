from langchain_openai import ChatOpenAI
from config import ARK_API_KEY
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional


from langchain_core.output_parsers import JsonOutputParser
from config import API_URL, ARK_API_KEY


llm = ChatOpenAI(
    temperature=0,
    openai_api_base= API_URL,
    openai_api_key= ARK_API_KEY,	# app_key
    model_name="deepseek-v3-241226",	# 模型名称
)


parser = JsonOutputParser()
response = parser.invoke(llm.invoke("生成JSON格式的天气预报"))
print(response)


