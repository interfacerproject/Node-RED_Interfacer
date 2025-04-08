import ast
import inspect
from pathlib import Path

TAG = "expose_as"

def extract_interface_functions(source_code):
    tree = ast.parse(source_code)
    lines = source_code.splitlines()
    functions = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            comment_line = lines[node.lineno - 2].strip()
            if comment_line.startswith(f"# {TAG}"):
                parts = comment_line.split()
                method = parts[2] if len(parts) > 2 else "post"
                functions.append((node, method))
    return functions

def get_type(t):
    return ast.unparse(t) if t else "Any"

def get_docstring(fn_node):
    return ast.get_docstring(fn_node) or ""

def build_fastapi_code(functions, filename):
    code = [
        "from fastapi import FastAPI",
        "from pydantic import BaseModel, Field",
        "from typing import Any, Optional, List, Dict",
        f"import {filename}",
        "",
        "app = FastAPI()",
        "",
    ]

    for fn, method in functions:
        name = fn.name
        # breakpoint()
        docstring = repr(get_docstring(fn))
        args = [(a.arg, get_type(a.annotation), a) for a in fn.args.args]
        returns = get_type(fn.returns)

        # Create input model for POST
        if method == "post":
            code.append(f"class {name.title()}Input(BaseModel):")
            for arg_name, arg_type, arg_obj in args:
                default = (
                    f" = {ast.unparse(arg_obj.default)}"
                    if hasattr(arg_obj, "default") and not isinstance(arg_obj.default, ast.Constant)
                    else ""
                )
                if default or arg_obj.arg != "self":
                    code.append(f"    {arg_name}: {arg_type}{default}")
            code.append("")

        # Output model
        code.append(f"class {name.title()}Output(BaseModel):")
        code.append(f"    result: {returns}\n")

        # API route
        route = f"@app.{method}('/{name}', response_model={name.title()}Output)"
        code.append(route)

        # Function signature
        if method == "get":
            params = ", ".join([
                f"{arg}: {typ}"
                for arg, typ, _ in args
                if arg != "self"
            ])
            call_args = ", ".join([arg for arg, _, _ in args])
            code.append(f"def {name}_endpoint({params}):")
        else:
            call_args = ", ".join([f"input.{arg}" for arg, _, _ in args])
            code.append(f"def {name}_endpoint(input: {name.title()}Input):")

        # Docstring
        code.append(f"    ''{docstring}''")
        code.append(f"    result = {filename}.{name}({call_args})")
        code.append(f"    return {{'result': result}}\n")

    return "\n".join(code)

def main(input:str, output:str):
    # breakpoint()
    source_code = Path(input).read_text()
    functions = extract_interface_functions(source_code)
    fastapi_code = build_fastapi_code(functions, Path(input).stem)
    Path(output).write_text(fastapi_code)
    print(f"✅ FastAPI wrapper generated in {output}")


if __name__ == "__main__":
    import argparse    
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '-i', '--input',
        dest='input',
        action='store',
        required=True,
        help='specifies the name of the file containing the functions',
    )
    parser.add_argument(
        '-o', '--output',
        dest='output',
        action='store',
        required=True,
        help='specifies the name of the file to write',
    )
    args, unknown = parser.parse_known_args()

    if len(unknown) > 0:
        print(f'Unknown options {unknown}')
        parser.print_help()
        exit(-1)

    main(args.input, args.output)
