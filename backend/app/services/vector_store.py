from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from app.services.embedder import embed_text

def get_similar_chunks(texts, query_embedding, top_k=5):
    embeddings = [embed_text(t) for t in texts]
    similarities = cosine_similarity([query_embedding], embeddings)[0]

    sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    top_indices = sorted_indices[:top_k]

    return [{"text": texts[i], "score": similarities[i]} for i in top_indices]