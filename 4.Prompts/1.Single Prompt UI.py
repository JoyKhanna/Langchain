import streamlit
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace
import streamlit as st

load_dotenv()

model = ChatOpenAI(model = "gpt-4o-mini")

st.header("Research Tool")
prompt = st.text_input("Enter your prompt")

if st.button("Summarize") :
    result = model.invoke(prompt)
    print(result)
    st.write(result)
