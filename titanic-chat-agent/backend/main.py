"""FastAPI entrypoint for the Titanic chat agent backend."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.agent import answer_question

app = FastAPI(title="Titanic Chat Agent")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="Natural language user question")


class ChatResponse(BaseModel):
    response: str
    tool_used: str | None = None
    visualization_base64: str | None = None


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a natural-language question about Titanic data."""
    try:
        result = answer_question(request.question)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ChatResponse(**result)

