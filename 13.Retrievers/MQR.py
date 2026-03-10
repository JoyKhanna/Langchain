# Multiquery Retriever
#   Sometimes the user query isnt very defined, so we use MQR, which takes the multiple meaning the query could mean and then gives results accordingly
#   It essentially finds more relevant document
#   MMR removes similar documents


from dotenv import load_dotenv
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

docs = [Document(page_content = "Regular walking boosts heart health and can reduce symptoms of depression."),
        Document(page_content = "Consuming leafy greens and fruits helps detox the body and improve longevity."),
        Document(page_content = "Deep sleep is crucial for cellular repair and emotional regulation."),
        Document(page_content = "Mindfulness and controlled breathing lower cortisol and improve mental clarity."),
        Document(page_content = "Drinking sufficient water throughout the day helps maintain metabolism and energy."),
        Document(page_content = "The solar energy system in modern homes helps balance electricity demand."),
        Document(page_content = "Python balances readability with power, making it a popular system design language."),
        Document(page_content = "Photosynthesis enables plants to produce energy by converting sunlight."),
        Document(page_content = "The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement."),
        Document(page_content = "Black holes bend spacetime and store immense gravitational energy.")]

embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small")

vector_store = FAISS.from_documents(documents = docs,
                                    embedding = embedding_model)

similarity_retriever = vector_store.as_retriever(search_type = "similarity",
                                      search_kwargs = {"k" : 5})

multiquery_retriever = MultiQueryRetriever.from_llm(retriever = vector_store.as_retriever(search_kwargs = {"k" : 5}),
                                                    llm = ChatOpenAI(model = "gpt-4o-mini"))

query = "How to improve energy levels and maintain balance ?"

similarity_result = similarity_retriever.invoke(query)
multiquery_result = multiquery_retriever.invoke(query)

for i, doc in enumerate(similarity_result):
    print(f"Result {i + 1} :")
    print(doc.page_content)


for i, doc in enumerate(multiquery_result):
    print(f"Result {i + 1} : ")
    print(doc.page_content)

