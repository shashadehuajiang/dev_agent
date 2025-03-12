# -*- coding: utf-8 -*-
import sys
import uuid
import os
import re
import string
import json
from datetime import datetime
from typing import Optional, List, Dict, Union
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import API_URL, ARK_API_KEY, API_MODEL_NAME
from openmanus.mylib import run_agent

# 初始化语言模型
llm = ChatOpenAI(
    openai_api_base=API_URL,
    openai_api_key=ARK_API_KEY,
    model_name=API_MODEL_NAME
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
        description="依赖的其他文件路径"
    )

class TaskNode(BaseModel):
    """统一的任务节点模型"""
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="节点唯一标识"
    )
    description: str = Field(..., description="任务描述")
    folder_name: Optional[str] = Field(
        default=None,
        description="由模型生成的文件夹名称"
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="父节点ID（根节点为空）"
    )
    children: List["TaskNode"] = Field(
        default_factory=list,
        description="子任务节点列表"
    )
    file_specs: List[FileSpec] = Field(
        default_factory=list,
        description="需要生成的文件规范"
    )
    generated_files: List[str] = Field(
        default_factory=list,
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
        self.max_depth = max_depth  # 最大递归深度限制

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
    
        if current_depth >= self.max_depth:
            self._print_process(f"⚠️ 达到最大深度限制 {self.max_depth}，停止分解")
            node.status = "completed"
            return
    
        node.status = "processing"
        self.indent_level += 1
        self._print_process(f"处理节点 [{node.id[:8]}]（深度 {current_depth}）：{node.description}")
        
        response = await self._analyze_requirement(node)
        node.folder_name = response.get('folder_name', f"node_{node.id[:6]}")  # 设置文件夹名称
        
        if response.get('type') == 'direct':
            node.file_specs = [FileSpec(**f) for f in response['files']]
        else:
            # 先处理子任务
            for subtask in response.get('subtasks', []):
                child = TaskNode(
                    description=subtask['description'],
                    folder_name=subtask.get('folder_name'),
                    parent_id=node.id,
                )
                if 'files' in subtask:
                    child.file_specs = [FileSpec(**f) for f in subtask['files']]
                node.children.append(child)
                self.node_map[child.id] = child
                await self._process_node(child, current_depth + 1)
            
            # 处理当前节点文件（在子节点之后处理以实现依赖解析）
            if 'files' in response:
                node.file_specs = [FileSpec(**f) for f in response['files']]
    
        await self._generate_files(node)
        node.status = "completed" if all(f in node.generated_files for f in self._get_expected_files(node)) else "failed"
        self.indent_level -= 1

    async def _analyze_requirement(self, node: TaskNode) -> Dict:
        """分析用户需求生成实现方案"""
        parent_path = self._get_node_path(node.parent_id) if node.parent_id else ""
        
        prompt_template = ChatPromptTemplate.from_template(
            """作为全栈开发专家，分析需求并规划文件结构。遵循以下原则：
1. 单个节点最多生成5个核心文件
2. 父节点负责框架，子节点处理具体模块
3. 资源文件集中放在resources目录
4. 确保文件路径符合当前节点位置：{current_path}
5. 为每个节点生成简洁的英文文件夹名（使用小写字母和下划线）
6. 用JSON格式返回，包含：
- type: direct（直接实现）或 split（需要拆分）
- folder_name: 当前节点的文件夹名称
- files（当前节点需要生成的文件列表）
- subtasks（需要拆分的子任务列表）

返回示例：
{{
    "type": "split",
    "folder_name": "flappy_bird_main",
    "files": [
        {{
            "file_type": "code",
            "file_name": "main.py",
            "purpose": "程序入口",
            "template": "import pygame\\n..."
        }}
    ],
    "subtasks": [
        {{
            "description": "实现游戏角色系统",
            "folder_name": "character_system",
            "files": [
                {{
                    "file_type": "code", 
                    "file_name": "character.py",
                    "purpose": "角色类定义"
                }}
            ]
        }}
    ]
}}

当前节点任务：{requirement}"""
        )
        
        messages = prompt_template.format_messages(
            requirement=node.description,
            current_path=self._get_node_path(node.id)
        )
        
        response = await llm.ainvoke(messages)
        self._log_api_call(messages, response.content)
        return parser.invoke(response)

    def _get_expected_files(self, node: TaskNode) -> List[str]:
        """获取节点预期生成的文件路径"""
        return [os.path.join(self._get_node_path(node.id), f.file_name) 
                for f in node.file_specs]

    async def _generate_files(self, node: TaskNode):
        """生成节点关联的所有文件"""
        node_dir = self._get_node_path(node.id)
        os.makedirs(node_dir, exist_ok=True)
        
        for file_spec in node.file_specs:
            file_path = os.path.join(node_dir, file_spec.file_name)
            if os.path.exists(file_path):
                continue
                
            context = {
                "node_description": node.description,
                "file_spec": file_spec.dict(),
                "dependencies": [
                    os.path.join(self._get_node_path(node.parent_id), dep)
                    for dep in file_spec.dependencies
                ],
                "save_path": file_path
            }
            
            gen_success = await run_agent(json.dumps(context), max_steps=50)
            if gen_success:
                node.generated_files.append(file_path)
                self._print_process(f"📄 生成文件：{file_path}")
            else:
                self._print_process(f"❌ 文件生成失败：{file_spec.file_name}")

    def _get_node_path(self, node_id: str) -> str:
        """获取节点对应的文件路径"""
        path_parts = []
        current_node = self.node_map.get(node_id)
        
        while current_node:
            if current_node.folder_name:
                # 清理非法字符并生成有效文件夹名称
                valid_name = re.sub(r'[^a-z0-9_-]', '', current_node.folder_name.lower())
                if not valid_name:
                    valid_name = f"node_{current_node.id[:6]}"
                path_parts.append(valid_name)
            else:
                path_parts.append(f"node_{current_node.id[:6]}")
            current_node = self.node_map.get(current_node.parent_id) if current_node.parent_id else None
        
        path_parts.reverse()
        return os.path.join("generated", *path_parts)

async def main():
    from gen_openmanus_config import init_config
    init_config()
    generator = UnifiedFileGenerator(max_depth=3)
    task_tree = await generator.build_tree(
        "写个python程序，输出0到100"
    )
    print("\n生成文件列表：")
    for f in task_tree.generated_files:
        print(f" - {f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

