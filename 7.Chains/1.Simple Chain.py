from docutils.nodes import description
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI(model = "gpt-4o-mini")

template = PromptTemplate(template = "Give 2 facts about {topic}",
                        input_variables = ["topic"])

class Facts(BaseModel):
    fact1 : str = Field(description = "First Fact")
    fact2 : str = Field(description = "Second Fact")
    fact3 : str = Field(description = "Third Fact")

structured_model = model.with_structured_output(Facts)

chain = template | structured_model

response = chain.invoke({"topic" : "white hole"})

print(type(response))
print("\n\n")
print(response)
print("\n\n")
print(response.fact1)
print(response.fact2)
print(response.fact3)

chain.get_graph().print_ascii()