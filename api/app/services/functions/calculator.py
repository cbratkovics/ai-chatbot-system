import ast
import operator
import math
from typing import Any, Dict, Union
from app.services.functions.base import BaseFunction, FunctionDefinition, FunctionParameter
import logging

logger = logging.getLogger(__name__)

class CalculatorFunction(BaseFunction):
    """Safe calculator function for mathematical operations"""
    
    def __init__(self):
        super().__init__()
        # Define safe operations
        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.FloorDiv: operator.floordiv,
        }
        
        self.functions = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'sqrt': math.sqrt,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'abs': abs,
            'round': round,
            'floor': math.floor,
            'ceil': math.ceil,
        }
        
        self.constants = {
            'pi': math.pi,
            'e': math.e,
        }
    
    def get_definition(self) -> FunctionDefinition:
        return FunctionDefinition(
            name="calculator",
            description="Perform mathematical calculations safely. Supports basic operations (+, -, *, /, **, %, //) and functions (sin, cos, tan, sqrt, log, exp, etc.)",
            parameters=[
                FunctionParameter(
                    name="expression",
                    type="string",
                    description="Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)')"
                )
            ],
            returns="number"
        )
    
    async def execute(self, expression: str) -> Union[float, int, str]:
        """Safely evaluate mathematical expression"""
        try:
            # Parse the expression into an AST
            tree = ast.parse(expression, mode='eval')
            
            # Evaluate the AST safely
            result = self._eval_node(tree.body)
            
            # Format result
            if isinstance(result, float):
                # Round to reasonable precision
                if result == int(result):
                    return int(result)
                else:
                    return round(result, 10)
            
            return result
            
        except ZeroDivisionError:
            return "Error: Division by zero"
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Calculator error: {str(e)}")
            return f"Error: Invalid expression - {str(e)}"
    
    def _eval_node(self, node):
        """Recursively evaluate AST nodes safely"""
        if isinstance(node, ast.Constant):
            return node.value
        
        elif isinstance(node, ast.Name):
            if node.id in self.constants:
                return self.constants[node.id]
            else:
                raise ValueError(f"Unknown variable: {node.id}")
        
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            elif isinstance(node.op, ast.USub):
                return -operand
            else:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_type = type(node.op)
            if op_type in self.operators:
                return self.operators[op_type](left, right)
            else:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
        
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in self.functions:
                    args = [self._eval_node(arg) for arg in node.args]
                    return self.functions[func_name](*args)
                else:
                    raise ValueError(f"Unknown function: {func_name}")
            else:
                raise ValueError("Complex function calls not supported")
        
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

class DataAnalysisFunction(BaseFunction):
    """Function for basic data analysis operations"""
    
    def get_definition(self) -> FunctionDefinition:
        return FunctionDefinition(
            name="data_analysis",
            description="Perform basic statistical analysis on numerical data",
            parameters=[
                FunctionParameter(
                    name="data",
                    type="array",
                    description="Array of numbers to analyze"
                ),
                FunctionParameter(
                    name="operation",
                    type="string",
                    description="Operation to perform",
                    enum=["mean", "median", "mode", "std", "variance", "min", "max", "sum", "count"]
                )
            ],
            returns="object"
        )
    
    async def execute(self, data: list, operation: str) -> Dict[str, Any]:
        """Perform statistical analysis"""
        import statistics
        
        if not data:
            return {"error": "No data provided"}
        
        try:
            # Ensure all data is numeric
            numeric_data = [float(x) for x in data]
            
            result = None
            if operation == "mean":
                result = statistics.mean(numeric_data)
            elif operation == "median":
                result = statistics.median(numeric_data)
            elif operation == "mode":
                try:
                    result = statistics.mode(numeric_data)
                except statistics.StatisticsError:
                    result = "No unique mode"
            elif operation == "std":
                result = statistics.stdev(numeric_data) if len(numeric_data) > 1 else 0
            elif operation == "variance":
                result = statistics.variance(numeric_data) if len(numeric_data) > 1 else 0
            elif operation == "min":
                result = min(numeric_data)
            elif operation == "max":
                result = max(numeric_data)
            elif operation == "sum":
                result = sum(numeric_data)
            elif operation == "count":
                result = len(numeric_data)
            
            return {
                "operation": operation,
                "result": result,
                "data_points": len(numeric_data)
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "operation": operation
            }