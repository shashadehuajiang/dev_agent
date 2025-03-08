from build_tree import UnifiedCodeGenerator

if __name__ == "__main__":
    generator = UnifiedCodeGenerator()
    task_tree = generator.build_tree(
        "随机生成1000位数字，要求数字中不包含1和3，保存为txt，随后读取txt，统计其中0的个数。必须分为两个子问题"
    )
    print(f"根节点代码路径：{task_tree.code_path}")

    