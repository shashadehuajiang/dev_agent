# -*- coding: utf-8 -*-
import uuid
import os
import re
import string
import json
import subprocess
from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import API_URL, ARK_API_KEY, API_MODEL_NAME

# Initialize language model
llm = ChatOpenAI(
    openai_api_base=API_URL,
    openai_api_key=ARK_API_KEY,
    model_name=API_MODEL_NAME
)
parser = JsonOutputParser()

# region Data Models
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
# endregion

class UnifiedCodeGenerator:
    """统一的代码生成器"""
    def __init__(self, max_depth: int = 3):
        self.root = None
        self.node_map = {}
        self.indent_level = 0
        self.generated_files = set()
        self.max_retries = 5
        self.max_depth = max_depth  # 最大递归深度限制

    # region Utility Methods
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
    # endregion

    # region Tree Construction
    def build_tree(self, requirement: str) -> TaskNode:
        """构建任务树"""
        self._print_process(f"构建任务树：'{requirement}'")
        self.root = TaskNode(description=requirement)
        self.node_map[self.root.id] = self.root
        self._process_node(self.root, current_depth=0)
        return self.root

    def _process_node(self, node: TaskNode, current_depth: int):
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
                self._process_node(child, current_depth + 1)  # 递归处理子节点

        # 生成代码并更新状态
        self._generate_code(node)
        node.status = "completed"
        self.indent_level -= 1
    # endregion

    # region Requirement Analysis
    def _analyze_requirement(self, requirement: str) -> Dict:
        """分析用户需求生成实现方案"""
        prompt_template = ChatPromptTemplate.from_template(
            """分析代码实现需求并返回：如果代码简单，可直接一个文件实现，则返回类信息，否则需拆分子任务进行解耦。
            子任务必须是python代码任务。
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
        response = llm.invoke(messages)
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
    # endregion

    # region Code Generation
    def _generate_code(self, node: TaskNode):
        """生成并测试代码"""
        if node.code_path and os.path.exists(node.code_path):
            return

        retry_count = 0
        previous_error = None
        current_code = None

        while retry_count < self.max_retries:
            # 生成或修复代码
            prompt_template = self._get_code_prompt(has_error=previous_error is not None)
            context = self._build_code_context(node)
            
            # 构建提示消息
            messages = prompt_template.format_messages(
                context=context,
                error=previous_error,
                code=current_code
            ) if previous_error else prompt_template.format_messages(context=context)

            # 调用模型生成代码
            response = llm.invoke(messages)
            self._log_api_call(messages, response.content)
            clean_code = self._clean_code(response.content)
            current_code = clean_code  # 保存当前生成的代码

            # 保存并测试代码
            self._save_code(node, clean_code)
            success, error = self._test_code(node.code_path)

            if success:
                self._print_process("代码运行测试通过")
                break  # 移除文档生成步骤
            else:
                retry_count += 1
                previous_error = error
                self._print_process(f"代码测试失败（尝试 {retry_count}/{self.max_retries}）")
                self._print_process(f"错误信息: {error}")
        else:
            self._print_process("⚠️ 达到最大重试次数，代码仍无法运行")

    def _get_code_prompt(self, has_error: bool) -> ChatPromptTemplate:
        """获取代码生成提示模板"""
        if has_error:
            return ChatPromptTemplate.from_template(
                """请修复以下代码错误，保持原始需求：
                [问题代码]:
                {code}

                [错误信息]:
                {error}

                原始需求:
                {context}

                修改要求:
                1. 检查测试assert内容是否正确
                2. 保持所有原始功能需求
                3. 确保符合PEP8规范
                4. 修复明显的语法错误和运行时错误
                5. 测试时不存在的文件需要先生成

                仅返回修正后的完整代码，不要省略:"""
            )
        else:
            return ChatPromptTemplate.from_template(
                """根据以下需求生成Python代码：
                要求：
                1. 包含完整的类实现和__main__测试块，print输入输出，使用assert断言
                2. 代码干净、无注释
                3. 优先使用单线程
                4. 如果使用多线程，必须包含多线程错误捕获并异常退出
                5. 必须使用子模块的API，严格按照API文档要求
                6. 第一行写utf-8编码

                参考信息：
                {context}

                仅返回一个完整代码块，不省略："""
            )

    def _build_code_context(self, node: TaskNode) -> str:
        """构建代码生成上下文信息"""
        context = []
        if node.class_info:
            # 添加详细的类信息
            context.append(f"## 主类定义: {node.class_info.name}")
            context.append(f"功能描述: {node.class_info.api_doc}\n")
            
            # 添加初始化方法信息
            context.append("### 初始化方法")
            context.append("def __init__(self):")
            context.append("    '''初始化实例'''\n")
            
            # 添加每个方法的详细信息
            context.append("### 功能方法列表")
            for func in node.class_info.functions:
                # 方法签名
                params = []
                for param in func.inputs:
                    param_desc = f"{param['name']}: {param['type']}"
                    if "description" in param:
                        param_desc += f" - {param['description']}"
                    params.append(param_desc)
                signature = f"{func.name}({', '.join(params)}) -> {func.output}"
                
                # 方法详细信息
                context.append(f"方法定义: {signature}")
                context.append(f"功能说明: {func.api_doc}")
                
                # 添加调用示例
                if func.api_doc and "示例：" in func.api_doc:
                    example_start = func.api_doc.index("示例：") + 3
                    example = func.api_doc[example_start:].split('\n')[0].strip()
                    context.append(f"调用示例: {example}")
                context.append("")  # 空行分隔

        # 添加子模块API引用信息
        if node.children:
            context.append("\n## 子模块API引用")
            parent_path_info = self._get_node_path_info(node)
            for child in node.children:
                child_path_info = self._get_node_path_info(child)
                relative_parts = child_path_info['path_parts'][len(parent_path_info['path_parts']):]
                relative_path = os.path.join(*relative_parts, child_path_info['filename'])
                
                # 添加子模块类信息
                if child.class_info:
                    context.append(f"### 来自文件: {relative_path}")
                    context.append(f"类名称: {child.class_info.name}")
                    context.append(f"功能描述: {child.class_info.api_doc}")
                    
                    # 添加子模块方法列表
                    method_list = []
                    for func in child.class_info.functions:
                        params = [f"{p['name']}:{p['type']}" for p in func.inputs]
                        method_desc = (
                            f"{func.name}({', '.join(params)}) -> {func.output} | "
                            f"{func.api_doc.split('.')[0]}"
                        )
                        method_list.append(method_desc)
                    context.append("可用方法:\n- " + "\n- ".join(method_list))
        
        return '\n'.join(context)

    def _clean_code(self, raw_code: str) -> str:
        """清理生成的代码"""
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
        """保存生成的代码到文件"""
        path_info = self._get_node_path_info(node)
        full_path = path_info['full_path']
        filepath = path_info['filepath']
        
        os.makedirs(full_path, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)

        node.code_path = filepath
        self.generated_files.add(filepath)
        self._print_process(f"生成文件：{filepath}")
        
        # 自动生成__init__.py文件
        if node.parent_id is not None:
            init_path = os.path.join(full_path, "__init__.py")
            if not os.path.exists(init_path):
                with open(init_path, 'w', encoding='utf-8') as f:
                    f.write("# Generated by UnifiedCodeGenerator\n")
                self._print_process(f"创建 __init__.py：{init_path}")

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
    
    def _test_code(self, filepath: str) -> (bool, str):
        """执行代码测试"""
        try:
            result = subprocess.run(
                ['python', filepath],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                return True, result.stdout
            else:
                error = result.stderr.strip() or result.stdout.strip()
                return False, error or "Unknown error"
        except subprocess.TimeoutExpired as e:
            return False, f"Timeout after 120 seconds: {e.stderr}"
        except Exception as e:
            return False, str(e)
    # endregion

if __name__ == "__main__":
    # 示例用法
    generator = UnifiedCodeGenerator(max_depth=2)
    task_tree = generator.build_tree(
        "设计并实现一个DNS隐蔽信道"
    )
    print(f"根节点代码路径：{task_tree.code_path}")

