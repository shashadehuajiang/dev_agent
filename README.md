# UnifiedCodeGenerator 代码生成器使用说明

## 简介
`UnifiedCodeGenerator` 是一个用于根据需求自动生成 Python 代码的工具。它能够将复杂的需求拆分成子任务，生成相应的代码。


## 安装依赖库
python 版本3.12

在运行代码之前，你需要安装所需的依赖库。可以使用以下命令通过 `pip` 进行安装：
```bash
pip install langchain-openai langchain-core pydantic
```

## 配置文件
在代码中，使用了 `config.py` 文件来配置 API 的相关信息，包括 `API_URL`、`ARK_API_KEY` 和 `API_MODEL_NAME`。你需要创建一个 `config.py` 文件，并按照以下格式填写相应的信息：
```python
API_URL = "your_api_url"
ARK_API_KEY = "your_api_key"
API_MODEL_NAME = "your_model_name"
```
请将 `your_api_url`、`your_api_key` 和 `your_model_name` 替换为你实际使用的 API 地址、API 密钥和模型名称。

[火山引擎API获取方式（支持deepseek系列）](https://zhuanlan.zhihu.com/p/23798747150)

## 运行代码
完成依赖库的安装和配置文件的设置后，你可以运行代码。在终端中，进入代码所在的目录，然后执行以下命令：
```bash
python your_script_name.py
```
请将 `your_script_name.py` 替换为实际的 Python 脚本文件名。

## 示例需求
代码中提供了一个示例需求，用于测试代码生成器的功能。该需求是“随机生成 1000 位数字，要求数字中不包含 1 和 3，保存为 txt，随后读取 txt，统计其中 0 的个数。必须分为两个子问题”。你可以根据需要修改 `__main__` 部分的需求内容，以测试不同的需求场景。

```python
if __name__ == "__main__":
    generator = UnifiedCodeGenerator()
    task_tree = generator.build_tree(
        "随机生成1000位数字，要求数字中不包含1和3，保存为txt，随后读取txt，统计其中0的个数。必须分为两个子问题"
    )
    print(f"根节点代码路径：{task_tree.code_path}")
```

## 注意事项
- 请确保你的 API 密钥和模型名称是有效的，否则代码可能无法正常调用 API 进行代码生成。
- 代码生成过程中可能会涉及到多次重试，以确保生成的代码能够通过测试。如果达到最大重试次数仍无法生成可运行的代码，会输出相应的错误信息。
- 生成的代码会保存在 `generated` 目录下，你可以查看生成的文件和 `__init__.py` 文件。