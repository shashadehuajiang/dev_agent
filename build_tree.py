# -*- coding: utf-8 -*-
import uuid
import os
import re
import json
from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import API_URL, ARK_API_KEY, API_MODEL_NAME
from openmanus.mylib import run_agent

# 记录用户prompt
global user_prompt
user_prompt = None

# 初始化语言模型
llm = ChatOpenAI(
    openai_api_base=API_URL,
    openai_api_key=ARK_API_KEY,
    model_name=API_MODEL_NAME,
    temperature=0.0
)
parser = JsonOutputParser()

# 数据模型
class FileSpec(BaseModel):
    """文件生成规范"""
    file_type: str = Field(..., description="文件类型（code/config/doc/resource/etc）")
    file_name: str = Field(..., description="文件名含后缀")
    purpose: str = Field(..., description="文件用途描述")
    template: Optional[str] = Field(
        default=None,
        description="文件内容模板或关键配置说明"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="依赖的其他文件路径（相对于根目录）"
    )

class TaskNode(BaseModel):
    """统一的任务节点模型"""
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="节点唯一标识"
    )
    description: str = Field(..., description="任务描述")
    parent_id: Optional[str] = Field(
        default=None,
        description="父节点ID（根节点为空）"
    )
    children: List["TaskNode"] = Field(
        default_factory=list,
        description="子任务节点列表"
    )
    file_spec: Optional[FileSpec] = Field(
        default=None,
        description="需要生成的文件规范"
    )
    generated_file: Optional[str] = Field(
        default=None,
        description="已生成的文件路径"
    )
    api_doc: Optional[str] = Field(
        default=None,
        description="节点功能文档（Markdown格式）"
    )
    status: str = Field(
        default="pending",
        description="任务状态: pending/processing/completed/failed"
    )

# 解决前向引用问题
TaskNode.model_rebuild()


class UnifiedFileGenerator:
    """统一文件生成器"""
    def __init__(self, max_depth: int = 5):
        self.root = None
        self.node_map = {}
        self.indent_level = 0
        self.max_depth = max_depth
        self.generated_files = set()


    def _print_process(self, message: str):
        """打印带缩进和时间戳的处理信息"""
        indent = "  " * self.indent_level
        print(f"{indent}🚀 {datetime.now().strftime('%H:%M:%S')} - {message}")

    def _log_api_call(self, request_messages, response_content):
        """记录API调用日志"""
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

    def _extract_functions_from_code(self, code: str) -> List[Dict]:
        """从代码模板中提取函数信息"""
        functions = []
        if not code:
            return functions
        
        try:
            pattern = re.compile(r'def\s+(\w+)\s*\((.*?)\)\s*(->\s*(\w+)\s*)?:', re.DOTALL)
            matches = pattern.findall(code)
            for match in matches:
                func_name = match[0]
                params_str = match[1].strip()
                return_type = match[3] if match[3] else None
                
                # 参数处理逻辑
                params = []
                if params_str:
                    param_list = re.split(r',\s*', params_str)
                    for param in param_list:
                        name_part = param.split(':')[0].split('=')[0].strip()
                        if name_part:
                            params.append(name_part)
                
                example_call = f"{func_name}({', '.join(params)})"
                functions.append({
                    "function_name": func_name,
                    "parameters": params_str,
                    "return_type": return_type,
                    "example_call": example_call
                })
        except Exception as e:
            self._print_process(f"⚠️ 解析函数时出错：{str(e)}")
        return functions

    async def build_tree(self, requirement: str) -> TaskNode:
        global user_prompt
        if user_prompt is None:
            user_prompt = requirement

        """构建任务树"""
        self._print_process(f"构建任务树：'{requirement}'")
        self.root = TaskNode(description=requirement)
        self.node_map[self.root.id] = self.root
        await self._process_node(self.root, current_depth=0)
        return self.root

    async def _process_node(self, node: TaskNode, current_depth: int):
        """递归处理任务节点"""
        if node.status == "completed":
            return
    
        if current_depth >= self.max_depth:
            self._print_process(f"⚠️ 达到最大深度限制 {self.max_depth}，停止分解")
            node.status = "completed"
            return
    
        node.status = "processing"
        self.indent_level += 1
        self._print_process(f"处理节点 [{node.id[:8]}]（深度 {current_depth}）：{node.description}")
        
        response = await self._analyze_requirement(node)
        
        if response.get('type') == 'direct':
            if 'file' in response:
                node.file_spec = FileSpec(**response['file'])
        else:
            for subtask in response.get('subtasks', []):
                child = TaskNode(
                    description=subtask['description'],
                    parent_id=node.id,
                )
                if 'file' in subtask:
                    child.file_spec = FileSpec(**subtask['file'])
                node.children.append(child)
                self.node_map[child.id] = child
                await self._process_node(child, current_depth + 1)
            
            if 'file' in response:
                node.file_spec = FileSpec(**response['file'])
    
        await self._generate_file(node)
        expected_file = self._get_expected_file(node)
        node.status = "completed" if node.generated_file and os.path.exists(expected_file) else "failed"
        self.indent_level -= 1

    async def _analyze_requirement(self, node: TaskNode) -> Dict:
        """分析用户需求生成实现方案"""        
        root_path = self._get_root_path()
        prompt_template = ChatPromptTemplate.from_template(
            """
当前任务：{requirement}

如果当前任务能一个文件写完，请不要拆分任务。    

结果用JSON格式返回，包含：
- type: direct（不拆分）或 split（需要拆分）
- file（当前节点需要生成的文件）
- subtasks（需要拆分的子任务列表）

遵循规则：
1. 每个节点只能生成1个file和最多3个subtasks
2. 父节点负责框架，串联模块，子节点处理具体模块
3. 为每个节点生成简洁的英文文件夹名（使用小写字母和下划线），不要包括{root_path}路径
4. 每个子任务的描述必须独立、精确、无歧义。
5. 对当前任务进行细化，放在description中

返回示例：
{{
    "type": "direct|split",
    "description": "详细描述",
    "file": {{
        "file_type": "",
        "file_name": "",
        "purpose": "",
        "interface": ""
    }},
    "subtasks": [
        {{
            "description": "详细描述",
            "file": {{
                "file_type": "code", 
                "file_name": "XXX.py",
                "purpose": "XXX"
            }}
        }}
    ]
}}

"""
        )
        
        messages = prompt_template.format_messages(
            requirement=node.description,
            root_path = root_path
        )
        
        response = await llm.ainvoke(messages)
        self._log_api_call(messages, response.content)
        return parser.invoke(response)

    def _get_expected_file(self, node: TaskNode) -> str:
        """获取节点预期生成的文件路径"""
        if node.file_spec:
            return os.path.join(self._get_root_path(), node.file_spec.file_name)
        return ""
    
    def _get_root_path(self) -> str:
        """获取项目根目录路径"""
        return "generated"

    async def _generate_file(self, node: TaskNode):
        if not node.file_spec:
            return
            
        root_dir = self._get_root_path()
        os.makedirs(root_dir, exist_ok=True)
        file_path = os.path.join(root_dir, node.file_spec.file_name)
        
        if os.path.exists(file_path):
            node.generated_file = file_path
            return
            
        # 收集可用代码模块信息
        available_modules = await self._collect_module_info(node)
        
        global user_prompt
        context = {
            "最终任务": user_prompt,
            "当前任务": node.description,
            "current_file": {
                "purpose": node.file_spec.purpose,
                "file_type": node.file_spec.file_type,
                "file_path": file_path,
                "dependencies": node.file_spec.dependencies
            },
            "available_modules": available_modules,
            "coding_rules": [
                "0. 请生成file_path文件，目的为purpose",
                "1. 符合purpose前提下，写的精炼",
                "2. 优先使用现有模块中的函数和类",
                "3. 使用正确的导入语句",
                "4. 返回完整代码，不省略",
                "5. 若缺素材自己想办法",
                "6. 代码路径在./" + root_dir,
                "7. 每次保存代码后必须测试代码",
                "8. 报错后必须修bug，修完继续保存代码测试",
            ]
        }
        
        gen_success = await run_agent(json.dumps(context), max_steps=50)
        if gen_success:
            node.generated_file = file_path
            self._print_process(f"📄 生成文件：{file_path}")
        else:
            self._print_process(f"❌ 文件生成失败：{file_path}")

    async def _collect_module_info(self, node: TaskNode) -> List[Dict]:
        """收集所有可用模块的详细信息"""
        modules = []

        # 收集子节点的模块
        for child in node.children:
            if child.file_spec:
                modules.append(self._parse_node_file(child))
        
        return modules

    def _parse_node_file(self, node: TaskNode) -> Dict:
        """解析节点文件生成模块信息"""
        if not node.file_spec or node.file_spec.file_type not in ["code"]:
            return {}
            
        module_data = {
            "module_path": node.file_spec.file_name,
            "purpose": node.file_spec.purpose,
            "classes": [],
            "functions": [],
        }
        
        if node.file_spec.template:
            module_data["functions"] = self._extract_functions_from_code(node.file_spec.template)
            module_data["classes"] = self._extract_classes_from_code(node.file_spec.template)

        return module_data

    def _extract_classes_from_code(self, code: str) -> List[Dict]:
        """从代码模板中提取类信息"""
        classes = []
        try:
            class_pattern = re.compile(
                r'^class\s+(\w+)\s*\(?(.*?)\)?:\n(.*?)(?=^\S|\Z)',
                re.MULTILINE | re.DOTALL
            )
            for match in class_pattern.findall(code):
                class_name, inheritance, class_body = match
                methods = []
                method_pattern = re.compile(
                    r'^\s+def\s+(\w+)\s*\((.*?)\)\s*(->\s*(\w+)\s*)?:',
                    re.MULTILINE | re.DOTALL
                )
                for method_match in method_pattern.findall(class_body):
                    method_name = method_match[0]
                    params = method_match[1].strip()
                    return_type = method_match[3] if method_match[3] else None
                    methods.append({
                        "method_name": method_name,
                        "parameters": params,
                        "return_type": return_type
                    })
                
                classes.append({
                    "class_name": class_name,
                    "inheritance": inheritance.strip(),
                    "public_methods": methods
                })
        except Exception as e:
            self._print_process(f"⚠️ 类解析错误：{str(e)}")
        return classes


async def main():
    from gen_openmanus_config import init_config
    init_config()
    generator = UnifiedFileGenerator(max_depth=5)
    prompt = "写一个坦克大战小游戏"
    task_tree = await generator.build_tree(
        prompt
    )
    print("\n生成文件列表：")
    if task_tree.generated_file:
        print(f" - {task_tree.generated_file}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


