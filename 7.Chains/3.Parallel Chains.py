from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model = "gpt-4o-mini")

templateFacts = PromptTemplate(template ="Give 2 facts about {topic}",
                               input_variables = ["topic"])

templateJokes = PromptTemplate(template ="Give 2 jokes about {topic}",
                               input_variables = ["topic"])

template3 = PromptTemplate(template = "Write the following paragraph in a better way : {facts} \n {jokes}",
                           input_variables = ["facts", "jokes"])

parallel_chain = RunnableParallel({"facts" : templateFacts | model | StrOutputParser(),
                                   "jokes" : templateJokes | model | StrOutputParser()})

merge_chain = parallel_chain | template3 | model | StrOutputParser()

response = merge_chain.invoke({"topic" : "AI"})

print(response)

merge_chain.get_graph().print_ascii()
