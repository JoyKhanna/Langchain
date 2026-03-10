from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal, Optional                          # what is typing

# Schema
class Review(BaseModel) :
    key_themes : list[str] = Field(description = "Mention the key themes in the review")
    summary : str = Field("Write a brief summary of the review of the product")
    sentiment : Literal["positive", "negative", "mixed", "neutral"] = Field(description = "Write the sentiment of the review, either negative or positive or mixed or neutral")
    pros : Optional[list[str]] = Field(description = "Pros mentioned in the review", default = None)
    cons : Optional[list[str]] = Field(description = "Cons mentioned in the review", default = None)
    name : Optional[str] = Field(description = "Name of the reviewer", default = None)

# but if something is optional then it means that, if the class doesn't have those values, even then it is fine, but how does the AI know that it is optional
# it will always respond naa unless we tell it that it is optional ???

load_dotenv()

model = ChatOpenAI(model = "gpt-4o-mini")
structured_model_output_pydantic = model.with_structured_output(Review)

# With structured output is actually making hidden parsers
response = structured_model_output_pydantic.invoke(""" The phone is good, but it gets heated too fast, and then the  performance drops like crazy.
                                                       Looking-wise, it is one of the best phones out in the market, but performance-wise its for regular users only.
                                                       If you are gaming on it, I would suggest buying the higher variant of it.
                                                       By Harsh Khanna""")

# Here we are getting name as really optional and getting none if not name is not there,
# but there even with gpt 5.2, the model printed name even when it wasn't mentioned                        ????

print(type(response))
print(response)
print(response.name)
