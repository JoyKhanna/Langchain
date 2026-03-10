from langchain_core.prompts import ChatPromptTemplate

domain = input("Enter the domain : ")
topic = input("Enter the topic : ")

chat_template = ChatPromptTemplate([
    ("system", "You are an helpful {domain} expert"),
    ("human", "Explain in simple terms, what is {topic}")
])

prompt = chat_template.invoke({"domain" : domain,
                               "topic" : topic})

print(prompt)