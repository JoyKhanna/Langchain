import streamlit
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model = "gpt-4o-mini")

streamlit.header("AI Research Helper")

paper_input = streamlit.selectbox("Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])
style_input = streamlit.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])
length_input = streamlit.selectbox("Select Length Of Explanation", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

template = load_prompt("F:\\GenAILangchain\\Prompts In Langchain\\template.json")

if streamlit.button("Summarize") :
    chain = template | model
    result = chain.invoke({"paper_input" : paper_input,
                           "length_input" : length_input,
                           "style_input" : style_input})

    streamlit.write(result.content)
