from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from openai import embeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sympy.polys.polyconfig import query

load_dotenv()

embedding = OpenAIEmbeddings(model = "text-embedding-3-large",
                              dimensions = 300)

document = ["Convolutional Neural Network (CNN). Best for images and videos; uses convolution layers to capture spatial features like edges and shapes. Widely used in face recognition, object detection, and medical imaging.",
            "Recurrent Neural Network (RNN). Designed for sequential data; remembers past information using hidden states. Used in text generation, time-series prediction, and speech recognition.",
            "Long Short-Term Memory (LSTM). A special RNN that handles long-term dependencies using gates to control memory flow. Common in machine translation, chatbots, and stock price prediction.",
            "Transformer. Uses self-attention instead of recurrence to model long-range dependencies efficiently. Powers modern LLMs like GPT, BERT, Gemini, and Claude.",
            "Autoencoder. Learns to compress data into a latent representation and reconstruct it back. Used for dimensionality reduction, anomaly detection, and denoising."]

query = "Tell me about transformer"

doc_embeddings = embedding.embed_documents(document)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]         # both should be 2-D list

index, score =  sorted(list(enumerate(scores)), key = lambda x: x[1])[-1]

print( )