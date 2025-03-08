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

llm = ChatOpenAI(
    openai_api_base=API_URL,
    openai_api_key=ARK_API_KEY,
    model_name=API_MODEL_NAME
)
parser = JsonOutputParser()

class FunctionInfo(BaseModel):
    """函数信息模型"""
    name: str
    inputs: List[Dict[str, str]]
    output: str
    api_doc: Optional[str] = Field(default=None, description="API文档说明")

class ClassInfo(BaseModel):
    """类信息模型"""
    name: str
    functions: List[FunctionInfo]
    api_doc: Optional[str] = Field(default=None, description="类级别API文档")

class TaskNode(BaseModel):
    """统一的任务节点模型"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    parent_id: Optional[str] = None
    children: List["TaskNode"] = Field(default_factory=list)
    class_info: Optional[ClassInfo] = None
    code_path: Optional[str] = None
    api_doc: Optional[str] = Field(default=None, description="节点功能文档")
    status: str = Field(default="pending", description="任务状态: pending/processing/completed")

TaskNode.model_rebuild()

class UnifiedCodeGenerator:
    """统一的代码生成器"""
    def __init__(self):
        self.root = None
        self.node_map = {}
        self.indent_level = 0
        self.generated_files = set()

    def _print_process(self, message: str):
        indent = "  " * self.indent_level
        print(f"{indent}🚀 {datetime.now().strftime('%H:%M:%S')} - {message}")

    def _log_api_call(self, request_messages, response_content):
        log_dir = "./log"
        os.makedirs(log_dir, exist_ok=True)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request": [{"content": msg.content} for msg in request_messages],
            "response": response_content
        }
        
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
        with open(os.path.join(log_dir, filename), "w", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)

    def build_tree(self, requirement: str) -> TaskNode:
        self._print_process(f"构建任务树：'{requirement}'")
        self.root = TaskNode(description=requirement)
        self.node_map[self.root.id] = self.root
        self._process_node(self.root)
        return self.root

    def _process_node(self, node: TaskNode):
        if node.status == "completed":
            return

        node.status = "processing"
        self.indent_level += 1
        self._print_process(f"处理节点 [{node.id[:8]}]：{node.description}")
        
        response = self._analyze_requirement(node.description)
        
        if response.get('type') == 'direct':
            node.class_info = self._parse_class_info(response['class'])
            node.api_doc = response['class'].get('api_doc')
        else:
            for subtask in response.get('subtasks', []):
                child = TaskNode(
                    description=subtask['description'],
                    parent_id=node.id,
                )
                if 'class' in subtask:
                    child.class_info = self._parse_class_info(subtask['class'])
                    child.api_doc = subtask['class'].get('api_doc')
                node.children.append(child)
                self.node_map[child.id] = child
                self._process_node(child)

        self._generate_code(node)
        node.status = "completed"
        self.indent_level -= 1

    def _analyze_requirement(self, requirement: str) -> Dict:
        prompt_template = ChatPromptTemplate.from_template(
            """分析需求并返回：如果可直接实现则返回类信息，否则拆分子任务
            返回格式（JSON）：
            {{
                "type": "direct|split",
                "class": {{
                    "name": "类名（直接实现时）",
                    "api_doc": "功能描述（包含输入输出说明）",
                    "functions": [
                        {{
                            "name":"方法名",
                            "input":"参数:类型,...",
                            "output":"返回类型",
                            "api_doc":"方法说明（含调用示例）"
                        }}
                    ]
                }},
                "subtasks": [
                    {{
                        "description": "子任务描述",
                        "class": {{...}}  // 可选
                    }}
                ]
            }}
            需求内容：{requirement}"""
        )
        messages = prompt_template.format_messages(requirement=requirement)
        response = llm.invoke(messages)
        self._log_api_call(messages, response.content)
        return parser.invoke(response)

    def _parse_class_info(self, data: Dict) -> ClassInfo:
        functions = []
        for func in data.get('functions', []):
            inputs = []
            for param in func.get('input', '').split(','):
                if param.strip() and ':' in param:
                    name, type_ = param.split(':', 1)
                    inputs.append({"name": name.strip(), "type": type_.strip()})
            functions.append(FunctionInfo(
                name=func['name'],
                inputs=inputs,
                output=func.get('output', ''),
                api_doc=func.get('api_doc')
            ))
        return ClassInfo(
            name=data['name'],
            functions=functions,
            api_doc=data.get('api_doc')
        )

    def _generate_code(self, node: TaskNode):
        if node.code_path and os.path.exists(node.code_path):
            return

        prompt_template = ChatPromptTemplate.from_template(
            """根据以下需求生成Python代码：
            要求：
            1. 包含完整的类实现和__main__测试块
            2. 代码精炼，不加注释
            3. 必须用子模块的API。
            4. PEP 8 规范
            5. import 语句必须在文件开头

            参考信息：
            {context}

            生成代码："""
        )

        context = []
        if node.class_info and node.class_info.functions:
            context.append(f"# 主类: {node.class_info.name}")
            context.append(f"'''{node.class_info.api_doc}'''")
            for func in node.class_info.functions:
                params = ', '.join([f"{p['name']}: {p['type']}" for p in func.inputs])
                context.append(f"方法: {func.name}({params}) -> {func.output}")

        if node.children:
            context.append("\n# 必须使用的API:")
            for child in node.children:
                context.append("\n")
                context.append(f"- 文件名: {child.code_path}")
                context.append(f"- API文档: {child.api_doc}")

        messages = prompt_template.format_messages(context='\n'.join(context))
        response = llm.invoke(messages)
        self._log_api_call(messages, response.content)
        
        clean_code = self._clean_code(response.content)
        self._save_code(node, clean_code)
        
        self._refine_api_documentation(node, clean_code)

    def _refine_api_documentation(self, node: TaskNode, code: str):
        prompt_template = ChatPromptTemplate.from_template(
            """根据代码生成详细的API文档，包含：
            1. 类功能描述和初始化参数
            2. 每个方法的参数说明、返回值、使用示例
            3. 模块的整体使用示例
            4. 子模块的调用说明（如果有）
            
            返回JSON格式：
            {{
                "class_doc": "类详细说明",
                "methods": [
                    {{
                        "name": "方法名",
                        "doc": "方法详细说明及示例"
                    }}
                ],
                "usage_example": "完整使用示例代码"
            }}
            
            代码：
            {code}"""
        )
        
        messages = prompt_template.format_messages(code=code)
        response = llm.invoke(messages)
        self._log_api_call(messages, response.content)
        
        try:
            doc_data = parser.invoke(response)
            if node.class_info:
                node.class_info.api_doc = doc_data.get("class_doc", "")
                for method_doc in doc_data.get("methods", []):
                    for func in node.class_info.functions:
                        if func.name == method_doc["name"]:
                            func.api_doc = method_doc["doc"]
                            break
            
            usage_example = doc_data.get("usage_example", "")
            node.api_doc = (
                f"## {node.class_info.name if node.class_info else 'Module'} Documentation\n"
                f"{doc_data.get('class_doc', '')}\n\n"
                "### Usage Example\n"
                f"```python\n{usage_example}\n```"
            )
        except Exception as e:
            self._print_process(f"生成API文档失败: {str(e)}")

    def _clean_code(self, raw_code: str) -> str:
        code_blocks = re.findall(r'```(?:python\s*)?\n(.*?)\n\s*```', raw_code, re.DOTALL)
        if code_blocks:
            cleaned = '\n\n'.join([block.strip() for block in code_blocks])
            cleaned = re.sub(r'# (Add your code here|TODO: Implement this)', '', cleaned)
            return cleaned
        
        code_lines = []
        in_code_block = False
        for line in raw_code.split('\n'):
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            if in_code_block or any(kw in line for kw in ['class ', 'def ', 'import ', 'from ', '@', 'if __name__']):
                code_lines.append(line)
        return '\n'.join(code_lines)

    def _save_code(self, node: TaskNode, code: str):
        # 生成目录结构基于节点层级
        path_parts = []
        current_node = node
        while current_node:
            if current_node.class_info:
                folder_name = re.sub(r'([A-Z])', r'_\1', current_node.class_info.name).lower().strip('_')
            else:
                valid_chars = f"-_{string.ascii_letters}{string.digits}"
                folder_name = ''.join(c for c in current_node.description[:20] if c in valid_chars) or f"node_{current_node.id[:6]}"
            path_parts.append(folder_name)
            current_node = self.node_map.get(current_node.parent_id) if current_node.parent_id else None

        # 反转路径顺序（从根到当前节点）
        path_parts.reverse()
        full_path = os.path.join("generated", *path_parts)
        
        # 确保文件名与目录名一致
        file_name = f"{path_parts[-1]}.py" if path_parts else "main.py"
        filepath = os.path.join(full_path, file_name)
        
        if filepath in self.generated_files:
            return
        
        os.makedirs(full_path, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
        node.code_path = filepath
        self.generated_files.add(filepath)
        self._print_process(f"生成文件：{filepath}")
        
        # 为非根节点添加 __init__.py
        if node.parent_id is not None:  # 非根节点
            init_path = os.path.join(full_path, "__init__.py")
            if not os.path.exists(init_path):
                with open(init_path, 'w', encoding='utf-8') as f:
                    f.write("# Generated by UnifiedCodeGenerator\n")
                self._print_process(f"创建 __init__.py：{init_path}")

    def _generate_filename(self, node: TaskNode) -> str:
        if node.class_info:
            name = re.sub(r'([A-Z])', r'_\1', node.class_info.name).lower().strip('_')
            return f"{name}.py"
        
        valid_chars = f"-_{string.ascii_letters}{string.digits}"
        filename = ''.join(c for c in node.description[:20] if c in valid_chars)
        return f"{filename}.py" if filename else f"module_{uuid.uuid4().hex[:6]}.py"

if __name__ == "__main__":
    generator = UnifiedCodeGenerator()
    task_tree = generator.build_tree(
        "随机生成1000位数字，要求数字中不包含1和3，保存为txt，随后读取txt，统计其中0的个数。分为两个子问题"
    )
    print(f"根节点代码路径：{task_tree.code_path}")

    