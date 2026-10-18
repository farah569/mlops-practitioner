import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from prodml.llm import ask_legal_question, get_db

logger = logging.getLogger("rag_api")
app = FastAPI(title="Egyptian Civil Code RAG API", version="1.0")


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_seconds: float


@app.get("/health")
def health_check():
    try:
        n = get_db()._collection.count()
    except Exception:
        logger.exception("vector store not available")
        raise HTTPException(status_code=503, detail="vector store not available")
    return {"status": "healthy", "documents_indexed": n}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    start = time.time()
    try:
        result = ask_legal_question(request.question)
    except Exception:
        logger.exception("ask failed")
        raise HTTPException(status_code=500, detail="Internal error")

    # keep relevance order, drop duplicates
    sources = list(dict.fromkeys(d.metadata["citation"] for d in result["context"]))
    return AnswerResponse(
        answer=result["answer"],
        sources=sources,
        latency_seconds=round(time.time() - start, 3),
    )
