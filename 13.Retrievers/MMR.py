# Maximum Marginal Relevance
#       In a query, we may get very relevant results, but those results can mean the same thing
#               So we wont get any different perspectives, all we get will be same info
#                       To solve this we have MMR, which fetches relevant results to the query which do not have same perspectives
#                       Or have less similarity btw results
#
# The similarity with the query should still be high.
# The similarity between the results should be low.
#
# High similarity(query, document)
# Low similarity(document_i, document_j)

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()

documents = [Document(page_content = "LangChain helps developers build LLM applications easily."),
             Document(page_content = "Chroma is a vector database optimized for LLM-based search."),
             Document(page_content = "Embeddings convert text into high-dimensional vectors."),
             Document(page_content = "OpenAI provides powerful embedding models")]


embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small")

vector_store = FAISS.from_documents(documents = documents,
                                    embedding = embedding_model)

retriever = vector_store.as_retriever(search_type = "mmr",
                                      search_kwargs = {"k" : 3,
                                                       "lambda_mult" : 0.5})                  # K : Top Results, lambda_mult : relevance-diversity balance
                                                                                            # lambda = 0, very diverse results even if not much relevant to query
                                                                                            # lambda = 1, most similar results to query, may be duplicate

query = "What is langchain ?"
result = retriever.invoke(query)

for i, doc in enumerate(result):
    print(f"Result {i + 1} :")
    print(doc.page_content)


# ENSEMBLE RETRIEVERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
















