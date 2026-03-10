from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(model = "meta-llama/Llama-3.1-8B-Instruct",
                          task = "text-generation")

HFmodel = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

template = PromptTemplate(template = "Give me the name, age and city of a fictional person\n {format_instruction}",
                          input_variables = [],          # if no input variable then why {format_instruction} in template, and why do we expect the name, age, city
                                                         # and also why is it telling that it needs the input, when no input is mentioned
                          partial_variables = {"format_instruction" : parser.get_format_instructions()})

#                         what is the diff. btw structured output format and this

chain = template | HFmodel | parser
response = chain.invoke({})
print(response)

#  as we tell how the LLM should output in pydantic object, so does it also remove the metadata ??