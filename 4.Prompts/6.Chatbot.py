from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model = "gpt-4o-mini")

chat_history = []
chat_history.append(SystemMessage(content = "You are a helpful AI assistant"))

while True :
    prompt = input("Enter prompt : ")
    chat_history.append(HumanMessage(content = prompt))

    if prompt == "exit" :
        break

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))

    print("AI : ", result.content)

print(chat_history)
