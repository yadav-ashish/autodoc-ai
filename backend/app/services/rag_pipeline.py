import os
import requests
from dotenv import load_dotenv
from app.services.embedder import embed_text
from app.services.vector_store import get_similar_chunks
from app.services.document_parser import extract_text_from_pdf

load_dotenv()

HF_API_TOKEN = os.getenv("<hf-api-token>")
HF_CHAT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
CHAT_API_URL = f"https://api-inference.huggingface.co/models/{HF_CHAT_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}


def format_prompt(chunks, question):
    context = "\n\n".join(chunk["text"] for chunk in chunks)
    prompt = f"Use the context below to answer the user's question.\n\nContext:\n{context}\n\nQuestion: {question}"
    return prompt


def answer_query(file_id: str, question: str) -> str:
    pdf_path = f"data/uploads/{file_id}.pdf"
    pages = extract_text_from_pdf(pdf_path)
    texts = [page["text"] for page in pages]

    query_embedding = embed_text(question)
    top_chunks = get_similar_chunks(texts, query_embedding, top_k=5)

    prompt = format_prompt(top_chunks, question)
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 300,
        }
    }

    response = requests.post(CHAT_API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    result = response.json()

    # Extract model response
    if isinstance(result, list) and "generated_text" in result[0]:
        return result[0]["generated_text"].split("Answer:")[-1].strip()
    elif isinstance(result, dict) and "generated_text" in result:
        return result["generated_text"].split("Answer:")[-1].strip()
    else:
        return str(result)
