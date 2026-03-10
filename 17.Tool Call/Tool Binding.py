import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
load_dotenv()

# Tool Making

@tool
def multiply(a: int, b: int) -> int:
    """Given 2 numbers a and b; this tool returns their product"""
    return a * b


print(multiply.invoke({"a": 3, "b": 5}))

# Tool Binding

llm = ChatOpenAI(model = "gpt-4o-mini")
llm_with_tools = llm.bind_tools([multiply])                                     # How does binding happen ????????????????

# print(llm_with_tools.invoke("Hi How are you ?").pretty_print())

# Tool Calling

chat_history = []
query = HumanMessage("A man ran 4 rounds of a track, which was of length 400m, what is the total distance he travelled ?")
chat_history.append(query)

response = llm_with_tools.invoke(chat_history)
print(response.pretty_print())                                          # It does the tool call here

chat_history.append(response)

tool_result = response.tool_calls[0]["name"].invoke(response.tool_calls[0])

chat_history.append(tool_result)

result = llm_with_tools.invoke(chat_history)
print(result.pretty_print())

print(chat_history)

