#   THEORY
import os
# We need structured output through LLMs
# But they usually give normal text in response
# Which is okay for humans to understand
# But it is not understandable for other systems
# If we give a sentiment analysis task, then it may review it in normal text
# But this won't be understandable by other systems, as it is plain text
# That's why we need structured output like JSON, or tabular data format


# Not all LLMs can give structured output
# To get structured output from them, we use with_structured_output() function
#       To get str. output we just need to input the format of our output in it
#               There are 3 types : TypedDict, JSON Schema, Pydantic

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict

load_dotenv()

model = ChatOpenAI(model = "gpt-4o-mini")

# Creating a schema for TypedDict

class Review(TypedDict) :                     # what is typeddict going in the class
    summary : str
    sentiment : str

structured_output_model = model.with_structured_output(Review)

dicti = {"summary" : "nice product",
         "sentiment" : "positive"}

obj1 = Review(summary = "nice product", sentiment = 67)
obj2 = Review(**dicti)

print(obj1)
print(obj2)

# Both are working, string and non string, cause TypedDict doesn't check things at runtime, for that we use Pydantic

# response = structured_output_model.invoke(""" The phone is good, but it gets heated too fast, and then the  performance drops like crazy.
#                                               Looking-wise, it is one of the best phones out in the market, but performance-wise its for regular users only.
#                                               If you are gaming on it, I would suggest buying the higher variant of it.""")


# print(response)                  # Output format is dict. just as review class as we defined


# How does the system know that it should be the summary of the text in "summary" key and sentiment in "sentiment" key
# With the with_structured_output function call, a system prompt is attached in the background that tells what to do according to our variables

# ##### BUT sentiment bas ek line ka hai ????? how does it know ki ek line ka hi ho


# Now lets say, you want to briefly describe what you want in those responses
# You can use Annotations

from typing import Annotated, Optional

class ReviewAnnotation(TypedDict) :
    keyThemes : Annotated[list[str], "Mention all the key themes discussed in the review in as list"]
    summary : Annotated[str, "A brief summary of the review"]
    sentiment : Annotated[str, "One word review, negative or positive or neural or mixed"]
    pros : Annotated[Optional[list[str]], "Write down all the pros written in the review"]
    cons : Annotated[Optional[list[str]], "Write down all the cons written in the review"]
    name : Annotated[Optional[str], "Write the name of the reviewer"]                                           # not working as expected

structured_output_model_annotated = model.with_structured_output(ReviewAnnotation)

# response = structured_output_model_annotated.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
#                                                        The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
#                                                        However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
#                                                        Pros:
#                                                            Insanely powerful processor (great for gaming and productivity)
#                                                            Stunning 200MP camera with incredible zoom capabilities """)

# print(response)

