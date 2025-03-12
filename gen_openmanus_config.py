import ast
import os
from pathlib import Path

def init_config():

    # 读取config.py内容
    config_path = 'config.py'
    with open(config_path, 'r', encoding='utf-8') as f:
        config_content = f.read()

    # 使用AST解析获取配置值
    class ConfigVisitor(ast.NodeVisitor):
        def __init__(self):
            self.config = {}

        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ['API_URL', 'ARK_API_KEY', 'API_MODEL_NAME']:
                    value = ast.literal_eval(node.value)
                    self.config[target.id] = value

    parsed_config = {}
    try:
        tree = ast.parse(config_content)
        visitor = ConfigVisitor()
        visitor.visit(tree)
        parsed_config = visitor.config
    except Exception as e:
        print(f'解析配置文件出错: {e}')
        exit(1)

    # 构建TOML内容
    toml_content = f'''
    # Global LLM configuration
    [llm]
    model = "{parsed_config.get('API_MODEL_NAME', '')}"
    base_url = "{parsed_config.get('API_URL', '')}"
    api_key = "{parsed_config.get('ARK_API_KEY', '')}"
    temperature = 0.2

    # Optional configuration for specific LLM models
    #[llm.vision]
    #model = "{parsed_config.get('API_MODEL_NAME', '')}"
    #base_url = "{parsed_config.get('API_URL', '')}"
    #api_key = "{parsed_config.get('ARK_API_KEY', '')}"
    '''

    # 写入目标文件
    target_path = Path('./openmanus/config/config.toml')
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(toml_content)
        print(f'配置已成功写入: {target_path.resolve()}')
    except Exception as e:
        print(f'写入配置文件出错: {e}')
        exit(1)

if __name__ == "__main__":
    init_config()