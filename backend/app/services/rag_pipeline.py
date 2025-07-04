import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.services.embedder import embed_text
from app.services.vector_store import get_similar_chunks
from app.services.document_parser import extract_text_from_pdf

# Load model once at startup
#MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1" ## Best for GPU.
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" ## Best for CPU.

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
model.to(device)
model.eval()

def format_prompt(chunks, question):
    context = "\n\n".join(chunk["text"] for chunk in chunks)
    prompt = f"<s>[INST] Use the context below to answer the user's question.\n\nContext:\n{context}\n\nQuestion: {question} [/INST]"
    return prompt

def answer_query(file_id: str, question: str) -> str:
    pdf_path = f"data/uploads/{file_id}.pdf"
    pages = extract_text_from_pdf(pdf_path)
    texts = [page["text"] for page in pages]

    query_embedding = embed_text(question)
    top_chunks = get_similar_chunks(texts, query_embedding, top_k=5)

    prompt = format_prompt(top_chunks, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()