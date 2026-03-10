from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model = "gpt-4o-mini")

template1 = PromptTemplate(template = "Give 10 line report about {topic}",
                           input_variables = ["topic"])

template2 = PromptTemplate(template = "Give a brief 3 line summary on the paragraph mentioned below\n {report}",
                           input_variables = ["report"])

chain = template1 | model | StrOutputParser() | template2 | model | StrOutputParser()

final_response = chain.invoke({"topic" : "Transformer models in AI"})

print(final_response)

chain.get_graph().print_ascii()
