import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prodml.llm import ask_legal_question

app = FastAPI(title="Egyptian Civil Code RAG API", version="1.0")


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_seconds: float


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start_time = time.time()
    try:
        result = ask_legal_question(request.question)

        sources = [doc.metadata["citation"] for doc in result["context"]]
        unique_sources = list(set(sources))

        latency = time.time() - start_time

        return AnswerResponse(
            answer=result["answer"], sources=unique_sources, latency_seconds=latency
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))
