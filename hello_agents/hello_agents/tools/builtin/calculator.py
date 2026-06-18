# my_calculator_tool.py
import ast
import operator
import math
from hello_agents.hello_agents.tools.registry import ToolRegistry
from hello_agents.hello_agents.tools.base import Tool, ToolParameter

def my_calculate(expression: str) -> str:
    """简单的数学计算函数"""
    if not expression.strip():
        return "计算表达式不能为空"

    # 支持的基本运算
    operators = {
        ast.Add: operator.add,      # +
        ast.Sub: operator.sub,      # -
        ast.Mult: operator.mul,     # *
        ast.Div: operator.truediv,  # /
    }

    # 支持的基本函数
    functions = {
        'sqrt': math.sqrt,
        'pi': math.pi,
    }

    try:
        node = ast.parse(expression, mode='eval')
        result = _eval_node(node.body, operators, functions)
        return str(result)
    except:
        return "计算失败，请检查表达式格式"

def _eval_node(node, operators, functions):
    """简化的表达式求值"""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left, operators, functions)
        right = _eval_node(node.right, operators, functions)
        op = operators.get(type(node.op))
        return op(left, right)
    elif isinstance(node, ast.Call):
        func_name = node.func.id
        if func_name in functions:
            args = [_eval_node(arg, operators, functions) for arg in node.args]
            return functions[func_name](*args)
    elif isinstance(node, ast.Name):
        if node.id in functions:
            return functions[node.id]

def create_calculator_registry():
    """创建包含计算器的工具注册表"""
    registry = ToolRegistry()

    # 注册计算器函数
    registry.register_function(
        name="my_calculator",
        description="简单的数学计算工具，支持基本运算(+,-,*,/)和sqrt函数",
        func=my_calculate
    )

    return registry


class CalculatorTool(Tool):
    """计算器工具类"""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="简单的数学计算工具，支持基本运算(+,-,*,/)和sqrt函数"
        )

    def run(self, parameters: dict) -> str:
        """执行计算"""
        expression = parameters.get("expression", "")
        print("000000000000========")
        return my_calculate(expression)

    def get_parameters(self):
        """获取参数定义"""
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="要计算的数学表达式，例如: 15 * 8 + 32"
            )
        ]


def create_calculator():
    """创建计算器工具实例"""
    return CalculatorTool()


# 为了向后兼容，创建别名
my_calculate_tool = create_calculator