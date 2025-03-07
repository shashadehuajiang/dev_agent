# -*- coding: utf-8 -*-
import uuid
import os
import re
import string
import json
from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import API_URL, ARK_API_KEY, API_MODEL_NAME

# 初始化大语言模型
llm = ChatOpenAI(
    openai_api_base=API_URL,
    openai_api_key=ARK_API_KEY,
    model_name=API_MODEL_NAME
)
parser = JsonOutputParser()


# 定义数据结构
class FunctionInfo(BaseModel):
    """函数信息模型"""
    name: str
    inputs: List[Dict[str, str]]  # [{"name": "param", "type": "str"}]
    output: str

class ClassInfo(BaseModel):
    """类信息模型"""
    name: str
    functions: List[FunctionInfo]

class TaskNode(BaseModel):
    """任务节点模型"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    is_leaf: bool = False
    parent_id: Optional[str] = None
    children: List["TaskNode"] = Field(default_factory=list)
    class_info: Optional[ClassInfo] = None
    code_path: Optional[str] = None

TaskNode.model_rebuild()  # 处理前向引用

class CodeGenerator:
    """代码生成器，负责构建任务树并生成代码"""
    def __init__(self):
        self.root = None
        self.node_map = {}
        # 添加缩进级别跟踪
        self.indent_level = 0

    def _print_process(self, message: str):
        """格式化输出处理过程"""
        indent = "  " * self.indent_level
        print(f"{indent}🚀 {datetime.now().strftime('%H:%M:%S')} - {message}")

    def _log_api_call(self, request_messages, response_content):
        """记录API调用日志"""
        log_dir = "./log"
        os.makedirs(log_dir, exist_ok=True)
        
        # 转换请求消息为可序列化格式
        serialized_request = []
        for msg in request_messages:
            serialized_request.append({
                "type": msg.type if hasattr(msg, "type") else type(msg).__name__,
                "content": msg.content
            })
        
        # 创建日志条目
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request": serialized_request,
            "response": response_content
        }
        
        # 生成唯一文件名
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
        filepath = os.path.join(log_dir, filename)
        
        # 写入文件
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)

    def build_tree(self, requirement: str) -> TaskNode:
        """构建任务树"""
        self._print_process(f"开始构建任务树，根需求：'{requirement}'")
        self.root = TaskNode(description=requirement)
        self._process_node(self.root)
        self._print_process("任务树构建完成")
        return self.root

    def _process_node(self, node: TaskNode):
        """递归处理任务节点"""
        self.indent_level += 1
        self._print_process(f"处理节点 [{node.id[:8]}]：{node.description}")
        
        response = self._split_requirement(node.description)
        
        if response.get('leaf', False):
            self._print_process("识别为叶子节点")
            node.is_leaf = True
            if 'class' in response:
                self._print_process(f"解析到类信息：{response['class']['name']}")
                node.class_info = self._parse_class_info(response['class'])
        else:
            self._print_process(f"需要拆分为 {len(response.get('subtasks', []))} 个子任务")
            for subtask in response.get('subtasks', []):
                child = TaskNode(
                    description=subtask['description'],
                    parent_id=node.id,
                    class_info=self._parse_class_info(subtask['class'])
                )
                node.children.append(child)
                self._process_node(child)
        
        # 后序遍历生成代码
        self._generate_code(node)
        self.indent_level -= 1

    def _split_requirement(self, requirement: str) -> Dict:
        """拆分需求接口"""
        prompt_template = ChatPromptTemplate.from_template(
            """请判断是否需要拆分该需求。若不需要，请返回：
            {{"leaf": true, "class": {{"name": "类名", "functions": [{{"name":"方法名", "input":"参数", "output":"类型"}}]}}}}
            若需要拆分，请返回：
            {{"subtasks": [{{"description": "子任务描述", "class": {{"name": "类名", "functions": [写清楚初始化方法、输入、输出个数、类型]}}}}], "leaf": false}}
            需求内容：{requirement}"""
        )
        messages = prompt_template.format_messages(requirement=requirement)
        # 调用API并记录日志
        response = llm.invoke(messages)
        self._log_api_call(messages, response.content)
        
        return parser.invoke(response)


    def _parse_class_info(self, data: Dict) -> ClassInfo:
        """解析类信息"""
        functions = []
        for func in data.get('functions', []):
            inputs = []
            for param in func.get('input', '').split(', '):
                if ':' in param:
                    name, type_ = param.split(':', 1)
                    inputs.append({"name": name.strip(), "type": type_.strip()})
            functions.append(FunctionInfo(
                name=func['name'],
                inputs=inputs,
                output=func.get('output', '')
            ))
        return ClassInfo(name=data['name'], functions=functions)

    def _generate_code(self, node: TaskNode):
        """代码生成逻辑"""
        self._print_process(f"生成代码 [{node.id[:8]}]（{'叶子节点' if node.is_leaf else '父节点'}）")
        if node.is_leaf:
            self._generate_leaf_code(node)
        else:
            self._generate_parent_code(node)


    def _clean_generated_code(self, raw_response: str) -> str:
        """清洗生成的代码，提取有效部分"""
        import re
        # 改进正则表达式以更灵活匹配代码块
        code_pattern = re.compile(
            r'```(?:python\s*)?\n(.*?)\n\s*```|# START CODE\s*(.*?)\s*# END CODE',
            re.DOTALL
        )
        matches = code_pattern.search(raw_response)
        
        if matches:
            # 提取第一个非空分组
            code = None
            for group in matches.groups():
                if group is not None:
                    code = group.strip()
                    break
            if code:
                return code
        
        # 如果未匹配到代码块，提取含有关键字的代码行
        code_lines = []
        in_code = False
        for line in raw_response.split('\n'):
            stripped_line = line.strip()
            if any(stripped_line.startswith(keyword) 
                for keyword in ['import', 'from', 'class', 'def', '@']):
                in_code = True
            if in_code:
                code_lines.append(line)
        return '\n'.join(code_lines).strip()
    

    def _generate_leaf_code(self, node: TaskNode):
        """生成叶子节点代码"""
        class_name = node.class_info.name
        self._print_process(f"生成叶子节点代码：{class_name}.py")

        prompt_template = ChatPromptTemplate.from_template(
            """请为以下类生成Python代码：
            类名：{class_name}
            方法列表：
            {methods}

            要求：
            1. 包含完整的类实现
            2. 在代码最后添加if __name__ == "__main__":块
            3. 在main块中编写测试用例，包含以下内容：
               - 创建类的实例
               - 调用主要方法
               - 使用断言验证结果
               - 打印测试通过信息
            4. 符合PEP8规范，仅输出代码，不要解释，英文注释"""
        )
        
        methods = []
        for func in node.class_info.functions:
            params = ', '.join([f"{p['name']}: {p['type']}" for p in func.inputs])
            methods.append(f"{func.name}({params}) -> {func.output}")
        
        messages = prompt_template.format_messages(
            class_name=node.class_info.name,
            methods='\n'.join(methods)
        )
        
        # 调用API并记录日志
        response = llm.invoke(messages)
        raw_code = response.content
        self._log_api_call(messages, raw_code)
        
        clean_code = self._clean_generated_code(raw_code)
        self._save_code(node, clean_code)

    def _generate_parent_code(self, node: TaskNode):
        """生成父节点整合代码（添加集成测试）"""
        self._print_process(f"整合 {len(node.children)} 个子节点代码")

        imports = []
        test_cases = []
        
        for child in node.children:
            if child.code_path:
                module_name = os.path.splitext(os.path.basename(child.code_path))[0]
                imports.append(f"from {module_name} import {child.class_info.name}")
                # 生成测试用例
                test_cases.append(f"""
    # 测试{child.class_info.name}功能
    try:
        obj_{child.class_info.name} = {child.class_info.name}()
        # 添加实际测试逻辑...
        print(f"{child.class_info.name} 测试通过")
    except Exception as e:
        print(f"{child.class_info.name} 测试失败: {{str(e)}}")
                """)

        prompt_template = ChatPromptTemplate.from_template(
            """请编写整合代码实现以下功能：
            需求描述：{description}

            需要整合的操作：
            {imports}

            要求：
            1. 实现完整的业务流程
            2. 添加if __name__ == "__main__":块
            3. 在main块中包含：
               - 所有子模块的功能测试
               - 完整的集成测试流程
               - 使用断言验证最终结果
               - 打印详细的测试报告
            4. 符合PEP8规范，仅输出代码，英文注释"""
        )
        
        messages = prompt_template.format_messages(
            description=node.description,
            imports='\n'.join(imports)
        )
        
        response = llm.invoke(messages)
        code = response.content
        self._log_api_call(messages, code)
        
        # 添加动态生成的测试用例
        raw_code = code.replace("# 添加测试用例在这里", "\n".join(test_cases))
        clean_code = self._clean_generated_code(raw_code)
        self._save_code(node, clean_code)

    def _generate_base_name(self, node: TaskNode) -> str:
        """生成简洁且有意义的基名"""
        # 优先使用类名中的核心词汇
        if node.class_info and node.class_info.name:
            class_name = node.class_info.name
            # 提取驼峰命名中的核心词汇（最多取3个）
            words = re.findall('[A-Z][a-z]+', class_name)
            if len(words) > 3:
                return f"{'_'.join(words[:2])}_{words[-1]}".lower()
            return '_'.join(words).lower()

        # 父节点使用关键动词+名词
        keywords = {
            'process': ['处理', '执行', '进行'],
            'manage': ['管理', '协调'],
            'integrate': ['整合', '组合']
        }
        
        # 提取描述中的核心词汇
        desc = node.description
        core_verbs = [k for k, v in keywords.items() if any(w in desc for w in v)]
        core_nouns = re.findall(r'\b(?:文件|数据|单词|统计|结果)\b', desc)
        
        # 生成组合名称（动词_名词1_名词2）
        name_parts = []
        if core_verbs:
            name_parts.append(core_verbs[0])
        name_parts += core_nouns[:2]
        
        # 保底策略
        if not name_parts:
            name_parts = ['system']
            
        return '_'.join(name_parts)

    def _get_unique_filename(self, base_name: str) -> str:
        """生成唯一文件名（最大长度25字符）"""
        base_name = base_name[:20]  # 保留扩展名空间
        ext = '.py'
        
        # 过滤非法字符
        valid_chars = f'-_.() {string.ascii_letters}{string.digits}'
        base_name = ''.join(c for c in base_name if c in valid_chars)
        
        counter = 1
        while True:
            suffix = f'_{counter}' if counter > 1 else ''
            filename = f"{base_name}{suffix}{ext}"
            full_path = os.path.join("generated", filename)
            
            if not os.path.exists(full_path) and len(filename) <= 25:
                return filename
            counter += 1

    def _save_code(self, node: TaskNode, code: str):
        """保存代码到文件"""
        base_name = self._generate_base_name(node)
        
        # 添加层级标记（根节点加main，中间节点加int）
        if node.parent_id is None:
            base_name += '_main'
        elif not node.is_leaf:
            base_name += '_int'
            
        filename = self._get_unique_filename(base_name)
        filepath = os.path.join("generated", filename)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
        node.code_path = filepath
        print(f"Saved: {filename} ({len(filename)} chars)")

# 使用示例
if __name__ == "__main__":
    generator = CodeGenerator()
    task_tree = generator.build_tree(
        "编写一个Python程序，读取文本文件，统计单词出现次数，并保存结果。分两个子任务"
    )
    print(f"根节点代码路径：{task_tree.code_path}")



