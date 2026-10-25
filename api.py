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

# Confidence gate: if even the best passage scores below this reranker score, the course probably does
# not cover the question, so we say so without calling the language model. -7.44 is the 5th percentile
# of top reranker scores on 200 in-course test questions (calibrated on held-out data), so about 95%
# of in-course questions pass; see README for the measured effect on out-of-course questions.
REFUSE_BELOW = float(os.environ.get("REFUSE_BELOW", "-7.44"))
NOT_COVERED = ("The course does not seem to cover this question. "
               "Try asking about a topic from the lectures, for example HTML, CSS or JavaScript basics.")

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


def is_covered(df) -> bool:
    """True if the best retrieved passage is relevant enough to answer from."""
    return float(df["rerank_score"].max()) >= REFUSE_BELOW
    
def build_prompt(question: str, df) -> str:
    df = df.copy()
    df["at"] = df["start"].apply(lambda s: f"{int(s // 60)}:{int(s % 60):02d}")
    chunks = df[["title", "number", "at", "text"]].to_json(orient="records")
    return (
        "You are a teaching assistant for a web development course. Below are transcript "
        "excerpts (video title, video number, time in minutes:seconds, text):\n\n"
        f"{chunks}\n---------------------------------\n"
        f"Student question: {question}\n\n"
        "Answer in 2 to 4 short sentences, speaking directly to the student. Use only the "
        "excerpts. First give the answer briefly, then say which video (number and title) "
        "and the minute:second to watch it. If the excerpts do not cover the question, say "
        "the course does not seem to cover it. Do not mention excerpts or chunks."
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
        if not is_covered(df):
            return Answer(answer=NOT_COVERED, sources=[], latency_seconds=round(time.time() - start, 2))
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
    
