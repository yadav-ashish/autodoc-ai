from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from app.services.rag_pipeline import answer_query

router = APIRouter()

class QueryRequest(BaseModel):
    file_id: str
    question: str

@router.post("/")
async def query_document(query: QueryRequest):
    try:
        response = answer_query(file_id=query.file_id, question=query.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
