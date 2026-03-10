from langchain_core.tools import tool


# Using Tool Decorator
@tool
def multiply(a: int, b: int) -> int:
    """Multiply 2 numbers"""
    return a * b

result = multiply.invoke({"a": 3, "b": 5})              # It's a runnable now
print(result)

print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.args_schema)
print(type(multiply))
print(multiply.args_schema.model_json_schema())


# Using Structured Tool

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyToolSchema(BaseModel):
    a: int = Field(description = "The first number to multiply")
    b: int = Field(description = "The second number to multiply")           # names of variables should be same in pydantic schema and function

def multiply(a: int, b: int) -> int:
    """Multiply 2 numbers"""
    return a * b

multiply_tool = StructuredTool.from_function(func = multiply,
                                             name = "multiply",
                                             description = "Multiply two numbers",
                                             args_schema = MultiplyToolSchema)

result = multiply_tool.invoke({"a": 3, "b": 4})
print(result)


#