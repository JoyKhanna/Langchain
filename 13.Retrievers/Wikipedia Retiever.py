# Retrievers fetch info from sources according to the query and return that info in form of document
#       They can be divided on basis of : Data Source     --> (Wikipedia Retriever, Vector Store, Archive Retriever)
#                                       : Search Strategy --> (Maximum Marginal Relevance, MultiQuery, Contextual)


# WikiPedia Retriever

from langchain_community.retrievers import WikipediaRetriever
from langchain_community.document_loaders import WikipediaLoader

retriever = WikipediaRetriever(top_k_results = 2)

docs = retriever.invoke("India Pakistan History")

print(type(docs))
print(len(docs))

print(docs[1].page_content)

# LOADER

loader = WikipediaLoader(
    query="India Pakistan history",
    load_max_docs=2
)

docs = loader.load()

print(len(docs))
print(docs[0].page_content)

