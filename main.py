from build_tree import UnifiedFileGenerator
from gen_openmanus_config import init_config

async def main():
    from gen_openmanus_config import init_config
    init_config()
    generator = UnifiedFileGenerator(max_depth=5)
    prompt = "写一个功能完善的坦克大战小游戏"
    task_tree = await generator.build_tree(
        prompt
    )
    print("\n生成文件列表：")
    if task_tree.generated_file:
        print(f" - {task_tree.generated_file}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())



