from build_tree import UnifiedCodeGenerator


prompt = """ 
使用langchain长会话聊天制作一个单机的文字冒险游戏，
游戏的内容基于三国时代。
玩家随机初始身份，加入某个阵营。
游戏是异步进行的，按照时间顺序进行，玩家可以在任何时刻打字进行游戏。
用户可以随时发出命令，游戏会做出响应。
每过一段时间会有一个事件发生，玩家可以自行操作，自由度非常高。
必须实现功能：用户可以使用控制台玩游戏，随时输入指令。

API_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_API_KEY = "c255abe3-80d2-4487-b524-1d439f967b16"
API_MODEL_NAME ="doubao-1-5-pro-32k-250115"
llm = ChatOpenAI(
    openai_api_base=API_URL,
    openai_api_key=ARK_API_KEY,
    model_name=API_MODEL_NAME
) 
"""

if __name__ == "__main__":
    generator = UnifiedCodeGenerator()
    task_tree = generator.build_tree(
        #"随机生成1000位数字，要求数字中不包含1和3，保存为txt，随后读取txt，统计其中0的个数。必须分为两个子问题"
        prompt
    )
    print(f"根节点代码路径：{task_tree.code_path}")

    