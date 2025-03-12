from build_tree import UnifiedFileGenerator
from gen_openmanus_config import init_config

async def main():
    init_config()
    generator = UnifiedFileGenerator(max_depth=5)
    task_tree = await generator.build_tree(
        "写篇计算机博士的论文，题目是隐蔽信道，要求10000字"
    )
    print("\n生成文件列表：")
    for f in task_tree.generated_files:
        print(f" - {f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


