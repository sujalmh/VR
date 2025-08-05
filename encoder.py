import streamlit as st
from sentence_transformers import SentenceTransformer

# Load the sentence transformer model once
@st.cache_resource
def get_sentence_transformer():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  # or any other pre-trained model you prefer

model = get_sentence_transformer()

# Cache for embeddings
@st.cache_resource
def get_embedding_cache():
    return {}

embedding_cache = get_embedding_cache()

def emb_text(client, text: str):
    if text in embedding_cache:
        return embedding_cache[text]
    else:
        # Use the sentence transformer model to encode the text
        embedding = client.encode(text)
        embedding_cache[text] = embedding
        return embedding
