from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

documents = [Document(page_content = "LangChain helps developers build LLM applications easily."),
             Document(page_content = "Chroma is a vector database optimized for LLM-based search."),
             Document(page_content = "Embeddings convert text into high-dimensional vectors."),
             Document(page_content = "OpenAI provides powerful embedding models."), ]

embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small")

vector_store = Chroma.from_documents(documents = documents,
                                     embedding = embedding_model,
                                     collection_name = "my_collection")

retriever = vector_store.as_retriever(search_kwargs = {"k": 2})

query = "How to build powerful embedding models ?"

result = retriever.invoke(query)

print(type(result))
print("\n\n")
print(result)
print("\n\n")

for i, doc in enumerate(result):
    print(f"Result {i + 1} :\n")
    print(doc.page_content)
