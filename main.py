from build_tree import UnifiedCodeGenerator
from gen_openmanus_config import init_config

async def main():
    init_config()
    generator = UnifiedCodeGenerator(max_depth=2)
    task_tree = await generator.build_tree(
        "写篇本科水平的计算机论文的pdf"
    )
    print(f"根节点代码路径：{task_tree.code_path}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


