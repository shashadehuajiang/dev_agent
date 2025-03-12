from build_tree import UnifiedCodeGenerator
from gen_openmanus_config import init_config

import sys
import os

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取openmanus所在目录
openmanus_dir = os.path.join(current_dir, 'openmanus')
# 将openmanus目录添加到sys.path中
sys.path.append(openmanus_dir)


async def main():
    init_config()
    generator = UnifiedCodeGenerator(max_depth=2)
    task_tree = await generator.build_tree(
        "写一个flappy bird游戏"
    )
    print(f"根节点代码路径：{task_tree.code_path}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


