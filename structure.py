import os
import ast

def get_directory_structure(directory: str) -> dict:
    """获取指定目录的结构信息，包括文件、函数调用信息及一级子目录的文件"""
    structure = {
        "current_directory": directory,
        "files": [],
        "subdirectories": []
    }
    if not os.path.exists(directory):
        return structure
    
    for entry in os.listdir(directory):
        entry_path = os.path.join(directory, entry)
        if os.path.isfile(entry_path):
            file_info = {
                "name": entry,
                "functions": []
            }
            if entry.endswith('.py'):
                try:
                    with open(entry_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    tree = ast.parse(code)
                    functions = []
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            args_str = ast.unparse(node.args)
                            output_str = "None"
                            for body_node in node.body:
                                if isinstance(body_node, ast.Return):
                                    if body_node.value is not None:
                                        output_str = ast.unparse(body_node.value)
                                        break
                            functions.append({
                                "name": node.name,
                                "inputs": args_str,
                                "output": output_str
                            })
                    file_info["functions"] = functions
                except Exception as e:
                    file_info["error"] = f"解析错误: {str(e)}"
            structure["files"].append(file_info)
        elif os.path.isdir(entry_path):
            subdir = {
                "name": entry,
                "files": []
            }
            for sub_entry in os.listdir(entry_path):
                sub_entry_path = os.path.join(entry_path, sub_entry)
                if os.path.isfile(sub_entry_path):
                    sub_file_info = {
                        "name": sub_entry,
                        "functions": []
                    }
                    if sub_entry.endswith('.py'):
                        try:
                            with open(sub_entry_path, 'r', encoding='utf-8') as f:
                                code = f.read()
                            tree = ast.parse(code)
                            functions = []
                            for node in ast.walk(tree):
                                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                    args_str = ast.unparse(node.args)
                                    output_str = "None"
                                    for body_node in node.body:
                                        if isinstance(body_node, ast.Return):
                                            if body_node.value is not None:
                                                output_str = ast.unparse(body_node.value)
                                                break
                                    functions.append({
                                        "name": node.name,
                                        "inputs": args_str,
                                        "output": output_str
                                    })
                            sub_file_info["functions"] = functions
                        except Exception as e:
                            sub_file_info["error"] = f"解析错误: {str(e)}"
                    subdir["files"].append(sub_file_info)
            structure["subdirectories"].append(subdir)
    
    return structure

