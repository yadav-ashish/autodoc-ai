from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import upload, query, summarize

app = FastAPI(
    title="AutoDoc AI",
    description="An intelligent document QA and summarization system using open-source GenAI.",
    version="0.1.0",
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set your frontend domain here in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include route modules
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(query.router, prefix="/query", tags=["Query"])
app.include_router(summarize.router, prefix="/summarize", tags=["Summarize"])


@app.get("/")
def read_root():
    return {"message": "AutoDoc AI backend is running."}
