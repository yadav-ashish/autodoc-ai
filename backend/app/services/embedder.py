from sentence_transformers import SentenceTransformer

# Load once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str):
    return embedding_model.encode([text])[0]