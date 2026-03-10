from pydantic import BaseModel, Field
from typing import Optional

class Review(BaseModel) :
    summary : str
    review_score : float = Field(gt = 0, lt = 10, default = 5, # greater than 0 and less than 10. Not equal
                                 description = "A decimal value representing the score of the review") # can also add regex expressions
    age : Optional[int] = None

# obj1 = Review(summary = "nice product", review_score = "negative")
# print(obj1)                                                                           # Will get type error

dicti = {"summary" : "good product",
         "review_score" : 8.7,
         "age" : 67}

obj2 = Review(**dicti)
print(type(obj2))            # gives __main__.Review which is BaseModel which is Pydantic
print(obj2)                  # Won't get type error

#  how does review_Score become optional if default value is given,

#   Converting Pydantic to Dict--------------------------------------------------------------------------------------------------

obj2_dict = obj2.model_dump()
print(type(obj2_dict))
print(obj2_dict["age"])

#   Converting Pydantic to JSON

obj2_JSON = obj2.model_dump_json()
print(type(obj2_JSON))
print(obj2_JSON)

