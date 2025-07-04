from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import uuid
from app.services.document_parser import extract_text_from_pdf

router = APIRouter()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Generate a unique filename
    file_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")

    # Save the file to disk
    with open(save_path, "wb") as f:
        f.write(await file.read())

    # Extract text
    parsed_pages = extract_text_from_pdf(save_path)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "num_pages": len(parsed_pages),
        "preview": parsed_pages[:2]  # send first 2 pages as preview
    }