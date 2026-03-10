# When you have a document which is not fully relevant with the query, but part of it is, then CCR, first fetches the
# relevant document and then compresses the document accordingly so that only the related part is considered

from dotenv import load_dotenv
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

docs = [Document(page_content = """The Grand Canyon is one of the most visited natural wonders in the world.
                                Photosynthesis is the process by which green plants convert sunlight into energy.
                                Millions of tourists travel to see it every year. The rocks date back millions of years."""),
        Document(page_content = """In medieval Europe, castles were built primarily for defense.
                                The chlorophyll in plant cells captures sunlight during photosynthesis.
                                Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""),
        Document(page_content = """Basketball was invented by Dr. James Naismith in the late 19th century.
                                It was originally played with a soccer ball and peach baskets. NBA is now a global league."""),
        Document(page_content = """The history of cinema began in the late 1800s. Silent films were the earliest form.
                                Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
                                Modern filmmaking involves complex CGI and sound design.""")]

embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small")

vector_store = FAISS.from_documents(documents = docs,
                                    embedding = embedding_model)

base_retriever = vector_store.as_retriever(search_kwargs = {"k" : 5})

llm = ChatOpenAI(model = "gpt-4o-mini")
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(base_retriever = base_retriever,
                                                       base_compressor = compressor)

query = "What is photosynthesis ?"

result = compression_retriever.invoke(query)

for i, doc in enumerate(result):
    print(f"Result {i + 1} : ")
    print(doc.page_content)


