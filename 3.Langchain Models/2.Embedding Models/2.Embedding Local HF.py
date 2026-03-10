from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")

text = "Delhi is capital of India"
vector = embedding.embed_query(text)            # How only one embedding vector ??? for the whole sentence
#                                                 is it cotext and meaning of whole sentence ??
#                                                 how to do it for words
print(str(vector))

# For Documents