# JSON schema is used when multiple languages are used in the project, like python in backend and javascript in frontend
from langchain_openai import ChatOpenAI

json_schema = {
      "title": "Review",
      "type": "object",
      "properties": {
        "key_themes": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Write down all the key themes discussed in the review in a list"
        },
        "summary": {
          "type": "string",
          "description": "A brief summary of the review"
        },
        "sentiment": {
          "type": "string",
          "enum": ["pos", "neg"],
          "description": "Return sentiment of the review either negative, positive or neutral"
        },
        "pros": {
          "type": ["array", "null"],
          "items": {
            "type": "string"
          },
          "description": "Write down all the pros inside a list"
        },
        "cons": {
          "type": ["array", "null"],
          "items": {
            "type": "string"
          },
          "description": "Write down all the cons inside a list"
        },
        "name": {
          "type": ["string", "null"],
          "description": "Write the name of the reviewer"
        }
      },
      "required": ["key_themes", "summary", "sentiment"]
}

model = ChatOpenAI(model = "gpt-4o-mini")
structured_output_model_json = model.with_structured_output(json_schema)

response = model.invoke("")
print(response)