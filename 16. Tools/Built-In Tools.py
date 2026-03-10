from langchain_community.tools import DuckDuckGoSearchRun

search_tools = DuckDuckGoSearchRun()

# results = search_tools.invoke("T20 News")
# print(results)
# How is this a tool ?? isnt this just a search query, for loading data from a source, just like yt loader and wikipedia loader


# SHELL TOOL

from langchain_community.tools import ShellTool

shell_tool = ShellTool()

result2 = shell_tool.invoke("python --version")
print(result2)