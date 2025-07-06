import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.services.document_parser import extract_text_from_pdf

# Reuse model
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1" ## Best for GPU.
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  ## Best for CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()


def summarize_document(file_id: str) -> str:
    pdf_path = f"data/uploads/{file_id}.pdf"
    pages = extract_text_from_pdf(pdf_path)
    full_text = "\n".join([page["text"] for page in pages])

    # Truncate if too long (TinyLlama max length is 2048)
    max_input_tokens = 1800
    inputs = tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=max_input_tokens
    )
    truncated_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    # Chat prompt format
    prompt = (
        f"<s>[INST] Summarize the following document:\n\n{truncated_text}\n\n[/INST]"
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only summary (after [/INST])
    if "[/INST]" in decoded:
        summary = decoded.split("[/INST]")[-1].strip()
    else:
        summary = decoded.strip()

    return summary
