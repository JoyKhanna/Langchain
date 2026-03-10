# Output parsers helps to convert
#       RAW LLM Response --> Structured Formats
#                            (JSON, CSV, Pydantic)
# They can be used with both kind of LLM models which can return structured output and which cannot
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
# StrOutputParser

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Our Flow :
#           Ask About Topic --> Get summary --> Give that summary --> Get brief summary

model = ChatOpenAI(model = "gpt-4o-mini")

template1 = PromptTemplate(template = "Write a brief line summary on the following text.\n {text}",
                           input_variables = ["text"])

template2 = template1

# response1 = model.invoke(template1.invoke({"text" : "black hole"}))
#
# response2 = model.invoke(template2.invoke({"text" : response1.content}))
#
# print(response1.content)
# print("\n\n\n")
# print(response2.content)

# print(type(response1))
# Now you put response.content in 2nd query
# Cause response contains metadata too, like "input_tokens", "output_tokens", "service_tier", etc.
# Which we do not want to send to our query
# This disrupts our chain so for that we need string parser, which extracts the response of our query and returns that

def parser(response):
    return response.content

# print(response1)
# print("\n\n")
# print(parser(response1))

chain = template1 | model | RunnableLambda(parser) | template1 | model | RunnableLambda(parser)

'''
final_response = chain.invoke({"text" : "white hole"})
print(final_response)
'''

parser2 = StrOutputParser()

chain2 = template1 | model | parser2 | template1 | model | parser2
# final_response2 = chain.invoke({"text" : "white hole"})
# print(final_response2)


llm = HuggingFaceEndpoint(repo_id = "meta-llama/Llama-3.1-8B-Instruct",
                          task = "text-generation")

HFmodel = ChatHuggingFace(llm = llm)

chain3 = template1 | HFmodel | parser2 | template1 | HFmodel | parser2

final_response_free = chain3.invoke({"text" : "white holes"})
print(final_response_free)