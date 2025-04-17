import ast
import inspect
from pathlib import Path
from typing import Any, Optional, List, Dict, Union
import textwrap

TAG = "expose_as"

class InterfaceFunction:
    file_to_read = None

    def __init__(self, node: ast.FunctionDef, method: str, file_to_read:str):
        self.node = node
        self.name = node.name
        self.args = node.args.args
        self.method = method
        InterfaceFunction.file_to_read = file_to_read
    
    def get_docstring(self):
        ret = ast.get_docstring(self.node) or ""
        return repr(ret)

    @staticmethod
    def resolve_type(t):
        if t is None:
            return "Any"
        t_str = ast.unparse(t)

        # Add support for nested dicts and lists
        if t_str == "dict":
            return "Dict[str, Any]"
        if t_str == "list":
            return "List[Any]"
        if t_str.startswith("Dict["):
            return t_str  # e.g., Dict[str, Dict[str, Any]]
        if t_str.startswith("List["):
            return t_str
        return t_str

    def generate_wrapper(self) -> str:
        arg_names = [arg.arg for arg in self.args if arg.arg != 'self']
        
        if self.method == 'post':    
            # args_unpack = ", ".join(f"{arg}=payload.{arg}" for arg in arg_names)
            args_copy = "\n    ".join(f"{arg} = copy.deepcopy(payload.{arg})" for arg in arg_names)
            return_keys = ", ".join(f'\"{arg}\": {arg}' for arg in arg_names)
            code = f"""
@app.{self.method.lower()}("/{self.name}")
def route_{self.name}(payload: {self.name.title()}Input):
    ''{self.get_docstring()}''
    import copy
    {args_copy}
    result = None
    try:
        result = {self.file_to_read}.{self.name}({', '.join(arg_names)})
    except Exception as e:
        return {{"error": str(e)}}
    return {{"result": result if result is not None else {{ {return_keys} }} }}
"""
        else:
            code = f"""
@app.{self.method.lower()}("/{self.name}")
def route_{self.name}(payload: {self.name.title()}Input):
    ''{self.get_docstring()}''
    result = None
    try:
        result = {self.file_to_read}.{self.name}({', '.join(arg_names)})
    except Exception as e:
        return {{"error": str(e)}}
    return {{"result": result}}
"""
        return code


def extract_interface_functions(file_to_read):
    """
        Parses the file and extracts function tagged for the interface
    """
    source_code = file_to_read.read_text()
    tree = ast.parse(source_code)
    lines = source_code.splitlines()
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            comment_line = lines[node.lineno - 2].strip()
            if comment_line.startswith(f"# {TAG}"):
                parts = comment_line.split()
                method = parts[2] if len(parts) > 2 else "post"
                functions.append(InterfaceFunction(node, method, file_to_read.stem))
    return functions

def generate_input_models(funcs: List[InterfaceFunction]) -> str:
    models = []
    for func in funcs:
        fields = [f"    {arg.arg}: Any" for arg in func.args if arg.arg != 'self']
        model = f"""
class {func.name.title()}Input(BaseModel):
{chr(10).join(fields)}
"""
        models.append(textwrap.dedent(model))
    return "\n".join(models)


def build_fastapi_code(functions):
    code = [
        "from fastapi import FastAPI",
        "from pydantic import BaseModel, Field",
        "from typing import Any, Optional, List, Dict, Union",
        f"import {InterfaceFunction.file_to_read}",
        "",
        "app = FastAPI()",
        "",
    ]
    code.append(generate_input_models(functions))

    for func in functions:
        wrapper = func.generate_wrapper()
        code.append(textwrap.dedent(wrapper))

    return "\n".join(code)
    

def main(input:str, output:str):
    # breakpoint()
    file_to_read = Path(input)
    if not file_to_read.exists():
        raise Exception(f"Cannot read file {input}")
    if 'wrapped' not in output:
        raise Exception(f"Refuse to overwrite file {output}")
    
    functions = extract_interface_functions(file_to_read)
    fastapi_code = build_fastapi_code(functions)
        

    wrapped_file = Path(output)
    wrapped_file.write_text(fastapi_code)
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


# if_lib
# generate_random_challenge, read_HMAC, read_keypair, get_id_person, get_location_id, \
# get_unit_id, get_resource_spec_id, get_resource, get_process, create_event, make_transfer, reduce_resource, set_user_location

