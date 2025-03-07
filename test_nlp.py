from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.output_parsers import JsonOutputParser
from config import API_URL, ARK_API_KEY, API_MODEL_NAME

# 初始化大语言模型
llm = ChatOpenAI(
    #temperature=0,
    openai_api_base=API_URL,
    openai_api_key=ARK_API_KEY,
    model_name=API_MODEL_NAME
)

# 初始化 JSON 输出解析器
parser = JsonOutputParser()

# 定义一个函数来处理用户输入并拆分问题
def split_programming_requirement(requirement):

    prompt = '请思考用户问题是否需要拆分为子问题。 \
    如果问题简单，不需要拆分，则返回json: {{leaf: True}}。   \
    如果问题复杂，则尝试拆分为可独立求解的子问题，以 JSON 格式输出。\
    子问题的描述需简洁且完备，可只看描述写出完整代码。\
    每个子问题都要包括一个类，类中有一或多个调用函数，明确类与函数的输入输出。     \
    格式如：{{stask_1: {{description:子问题1, class:子问题类1描述 }}，", stask_2: {{description:子问题2, class:子问题类2描述 }}...}} \
    类的描述例子：{{class: FileReader, functions: [{{name: read_file, input: file_path: str, output: content: str}}]}}    \
    ：{problem}' 

    # 创建提示模板
    prompt = ChatPromptTemplate.from_template(prompt)
    # 格式化提示消息
    formatted_prompt = prompt.format_messages(problem=requirement)
    # 调用大语言模型
    response = llm.invoke(formatted_prompt)
    # 解析响应为 JSON 格式
    parsed_response = parser.invoke(response)
    return parsed_response

# 示例用户输入
user_input = "编写一个 Python 程序，读取一个文本文件，统计文件中每个单词的出现次数，并将结果保存到另一个文本文件中。"

# 调用函数拆分问题
result = split_programming_requirement(user_input)

# 打印结果
print('type(result)', type(result))
print(result)



