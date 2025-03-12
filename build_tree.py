# -*- coding: utf-8 -*-
import sys
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
from openmanus.mylib import run_agent
import subprocess

# 初始化语言模型
llm = ChatOpenAI(
    openai_api_base=API_URL,
    openai_api_key=ARK_API_KEY,
    model_name=API_MODEL_NAME
)
parser = JsonOutputParser()

# 数据模型
class FunctionInfo(BaseModel):
    """函数信息模型"""
    name: str
    inputs: List[Dict[str, str]] = Field(..., description="输入参数列表，每个参数包含name和type")
    output: str = Field(..., description="返回值类型")
    api_doc: Optional[str] = Field(
        default=None,
        description="API文档说明（包含调用示例）"
    )

class ClassInfo(BaseModel):
    """类信息模型"""
    name: str = Field(..., description="类名称")
    functions: List[FunctionInfo] = Field(..., description="包含的方法列表")
    api_doc: Optional[str] = Field(
        default=None,
        description="类级别API文档（包含使用示例）"
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
    class_info: Optional[ClassInfo] = Field(
        default=None,
        description="关联的类信息"
    )
    code_path: Optional[str] = Field(
        default=None,
        description="生成代码文件路径"
    )
    api_doc: Optional[str] = Field(
        default=None,
        description="节点功能文档（Markdown格式）"
    )
    status: str = Field(
        default="pending",
        description="任务状态: pending/processing/completed"
    )

# 解决前向引用问题
TaskNode.model_rebuild()


class UnifiedCodeGenerator:
    """统一的代码生成器"""
    def __init__(self, max_depth: int = 5):
        self.root = None
        self.node_map = {}
        self.indent_level = 0
        self.generated_files = set()
        self.max_retries = 1
        self.max_depth = max_depth  # 最大递归深度限制

    # region 工具方法
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

    async def build_tree(self, requirement: str) -> TaskNode:
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
    
        # 深度限制检查
        if current_depth >= self.max_depth:
            self._print_process(f"⚠️ 达到最大深度限制 {self.max_depth}，停止分解")
            node.status = "completed"
            return
    
        node.status = "processing"
        self.indent_level += 1
        self._print_process(f"处理节点 [{node.id[:8]}]（深度 {current_depth}）：{node.description}")
        
        # 分析需求并生成子任务
        response = await self._analyze_requirement(node.description)
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
                await self._process_node(child, current_depth + 1)  # 递归处理子节点
    
        # 生成代码并更新状态
        await self._generate_code(node)
        node.status = "completed"
        self.indent_level -= 1

    async def _analyze_requirement(self, requirement: str) -> Dict:
        """分析用户需求生成实现方案"""
        prompt_template = ChatPromptTemplate.from_template(
            """分析需求并返回：如果需求简单，可直接一个文件实现，则返回类信息，否则需拆分子任务进行解耦。
            下面“本任务描述：”之后出现过的所有句子、代码必须包含在其中某一个子任务描述中，不得拆分后遗漏信息。
            子任务的描述必须完备，是完整详细的描述。
            必须严格定义所有函数的输入输出，包括初始化函数。
            返回格式（JSON）：
            {{
                "type": "direct|split",
                "class": {{
                    "name": "类名（直接实现时）",
                    "api_doc": "功能描述",
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
            本任务描述：{requirement}"""
        )
        messages = prompt_template.format_messages(requirement=requirement)
        response = await llm.ainvoke(messages)
        self._log_api_call(messages, response.content)
        return parser.invoke(response)

    def _parse_class_info(self, data: Dict) -> ClassInfo:
        """解析类信息"""
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

    async def _generate_code(self, node: TaskNode):
        """生成并测试代码"""
        if node.code_path and os.path.exists(node.code_path):
            return

        retry_count = 0
        previous_error = None
        current_code = None

        while retry_count < self.max_retries:
            # 生成或修复代码的上下文
            context = {
                "requirement": node.description,
                "class_info": node.class_info.model_dump() if node.class_info else None,  # 修复点1
                "error": previous_error,
                "existing_code": current_code,
                "save_path": self._get_node_path_info(node)["filepath"]
            }

            # 执行Agent流程
            gen_success = await run_agent(json.dumps(context), max_steps = 50)
            self._print_process("agent gen_success: , {gen_success}")
            
            # 测试代码
            code_path = self._get_node_path_info(node)["filepath"]
            success, error = self._test_code(code_path)  # 新增测试方法

            if gen_success and success:
                self._print_process("✅ 代码运行测试通过")
                node.code_path = code_path
                break
            else:
                retry_count += 1
                previous_error = error
                current_code = code_path
                self._print_process(f"⚠️ 代码测试失败（尝试 {retry_count}/{self.max_retries}）")
                self._print_process(f"错误信息: {error}")

        else:
            self._print_process("⛔ 达到最大重试次数，代码仍无法运行")
            node.status = "failed"
            return

        node.status = "completed"

    def _get_code_prompt(self, has_error: bool) -> ChatPromptTemplate:
        """获取代码生成提示模板"""
        # 提示模板内容保持不变...
        pass

    def _build_code_context(self, node: TaskNode) -> str:
        """构建代码生成上下文信息"""
        # 上下文构建逻辑保持不变...
        pass

    def _get_node_path_info(self, node: TaskNode) -> Dict:
        """获取节点对应的文件路径信息"""
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

        path_parts.reverse()
        full_path = os.path.join("generated", *path_parts)
        filename = f"{path_parts[-1]}.py" if path_parts else "main.py"
        filepath = os.path.join(full_path, filename)
        return {
            "path_parts": path_parts,
            "full_path": full_path,
            "filename": filename,
            "filepath": filepath
        }

    def _test_code(self, code_path: str) -> tuple:
        """测试生成的代码"""
        try:
            # 使用子进程执行代码测试
            result = subprocess.run(
                [sys.executable, code_path],
                check=True,
                capture_output=True,
                text=True,
                timeout=120
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, f"Exit code {e.returncode}\n{e.stderr}"
        except Exception as e:
            return False, str(e)

async def main():
    from gen_openmanus_config import init_config
    init_config()
    generator = UnifiedCodeGenerator(max_depth=2)
    task_tree = await generator.build_tree(
        "写一个flappy bird游戏"
    )
    print(f"根节点代码路径：{task_tree.code_path}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

