from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.summarizer import summarize_document

router = APIRouter()

class SummarizeRequest(BaseModel):
    file_id: str

@router.post("/")
async def summarize_file(request: SummarizeRequest):
    try:
        summary = summarize_document(file_id=request.file_id)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
