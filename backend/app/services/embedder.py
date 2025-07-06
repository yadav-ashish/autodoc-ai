import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("<hf-api-token>")
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMBEDDING_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}


def embed_text(text: str):
    response = requests.post(EMBEDDING_API_URL, headers=HEADERS, json={"inputs": text})
    response.raise_for_status()
    embeddings = response.json()

    # Average all token embeddings to get a single vector
    if isinstance(embeddings, list) and isinstance(embeddings[0], list):
        averaged = [sum(x) / len(x) for x in zip(*embeddings)]
        return averaged
    return embeddings
