"""
FastAPI service + small web page for the RAG teaching assistant.

Run (Ollama must be running with llama3.2 and bge-m3 pulled):
    uvicorn api:app --host 0.0.0.0 --port 8000
Then open http://localhost:8000 in a browser, or call the API:
    curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
         -d '{"question": "How do I create a link in HTML?"}'
"""
import os
import time
from typing import List

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.2")

app = FastAPI(title="RAG Teaching Assistant", version="1.0")


class Question(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(5, ge=1, le=10)


class Source(BaseModel):
    title: str
    number: str
    start: float
    end: float
    text: str


class Answer(BaseModel):
    answer: str
    sources: List[Source]
    latency_seconds: float


def retrieve(question: str, top_k: int):
    from hybrid_retrieval import hybrid_retrieve  # lazy: loads the index on first use

    return hybrid_retrieve(question, dense_k=20, sparse_k=20, fused_k=15, final_k=top_k)


def build_prompt(question: str, df) -> str:
    chunks = df[["title", "number", "start", "end", "text"]].to_json(orient="records")
    return (
        "I am teaching web development in a course. Here are video subtitle chunks "
        "(video title, video number, start and end time in seconds, text):\n\n"
        f"{chunks}\n---------------------------------\n"
        f'"{question}"\n'
        "The user asked this question. Answer in a human way: say in which video and at "
        "what timestamp the topic is taught and guide the user to that video. Do not mention "
        "the chunk format. If the question is unrelated to the course, say you can only "
        "answer questions about the course."
    )


def ask_llm(prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["response"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask(body: Question):
    start = time.time()
    try:
        df = retrieve(body.question, body.top_k)
        answer = ask_llm(build_prompt(body.question, df))
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Could not reach Ollama at {OLLAMA_URL}: {e}")
    sources = [
        Source(
            title=str(row["title"]),
            number=str(row["number"]),
            start=float(row["start"]),
            end=float(row["end"]),
            text=str(row["text"]),
        )
        for _, row in df.iterrows()
    ]
    return Answer(answer=answer, sources=sources, latency_seconds=round(time.time() - start, 2))


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))
