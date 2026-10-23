# RAG-Based AI Teaching Assistant
IEEE Published Research | Kaggle Top 10% | GATE 2025

Intelligent Q&A system for educational videos using Retrieval-Augmented Generation, with hybrid (dense + keyword) retrieval and cross-encoder reranking.

## Overview
Processes real lecture video content (7,252 chunks transcribed from 18 real videos via Whisper) into a queryable Q&A system. Given a question, it retrieves the most relevant lecture moments and answers with the exact video and timestamp.

## Architecture

```
Lecture videos -> FFmpeg audio extraction -> Whisper large-v2 (speech-to-text, Hindi->English translation)
        |
Text chunks with timestamps (7,252 real chunks)
        |
BGE-M3 embeddings (1024-dim) via Ollama
        |
   +----------------+----------------+
   |                                 |
FAISS IVF-Flat dense search    BM25 keyword search
   |                                 |
   +----------------+----------------+
                     |
       Reciprocal Rank Fusion (merges both ranked lists)
                     |
       Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
                     |
           LLaMA 3.2 (via Ollama) generates the answer
                     |
              Answer + video timestamp
```

## Measured Results

| Metric | Result | How it was measured |
|--------|--------|----------------------|
| Dataset | 7,252 real chunks, 18 lecture videos | Loaded directly from `embeddings.joblib` |
| FAISS search speed | **57x faster** than brute-force cosine similarity | Benchmarked: 11.95ms -> 0.21ms per query, same dataset |
| Embedding dimension | 1024 (BGE-M3) | Verified directly from the saved embedding vectors |
| Retrieval correctness | FAISS results match brute-force top-5 exactly in most queries; the rare 1-result difference is the expected recall/speed trade-off of approximate search | Verified by cross-checking both methods on multiple real queries |
| End-to-end query latency (retrieval + LLM generation) | **2.57s** steady-state (5.67s on first query, due to one-time model load) | Measured directly: `total_duration` from two real queries against the live pipeline |

## Why hybrid retrieval (FAISS + BM25 + reranking)
- **FAISS (dense/semantic)** catches meaning-based matches even with no shared words.
- **BM25 (sparse/keyword)** reliably catches exact technical terms (e.g. "z-index") that dense embeddings sometimes under-rank.
- **Reciprocal Rank Fusion** combines both ranked lists without needing to normalize incompatible score scales.
- **Cross-encoder reranking** re-scores the top ~15 fused candidates by reading the query and each chunk together, catching relationships separate-embedding search misses. Too slow to run on all 7,252 chunks, so it only runs on the shortlist - the standard "retrieve cheap, rerank precise" pattern.

## Tech Stack
- Speech-to-Text: OpenAI Whisper (large-v2)
- Embeddings: BGE-M3 (1024-dim) via Ollama
- Dense retrieval: FAISS (IVF-Flat index)
- Sparse retrieval: BM25 (rank_bm25)
- Fusion: Reciprocal Rank Fusion
- Reranking: cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence-transformers)
- LLM: LLaMA 3.2 via Ollama
- Processing: Python, Pandas, NumPy, Joblib

## Project Structure
```
rag-teaching-assistant/
├── jsons/                  # Real transcribed lecture chunks with timestamps
├── embeddings.joblib       # 7,252 real BGE-M3 embeddings (1024-dim)
├── faiss_index.bin         # Pre-built FAISS IVF-Flat index
├── video_to_mp3.py         # Step 1: video to audio
├── mp3_to_json.py          # Step 2: Whisper transcription
├── preprocess_json.py      # Step 3: generate embeddings
├── build_faiss_index.py    # Step 4: build the FAISS index
├── hybrid_retrieval.py     # FAISS + BM25 + RRF fusion + cross-encoder rerank
├── inference.py            # Step 5: ask a question, get an answer
├── evaluate_retrieval.py   # measure Recall@k / MRR for each retrieval setup
├── Dockerfile / docker-compose.yml   # run the app with Ollama in Docker
├── requirements.txt
└── README.md
```

## Quick Start
```bash
git clone https://github.com/Kaveri-Ghatage/rag-teaching-assistant.git
cd rag-teaching-assistant
pip install -r requirements.txt
ollama pull llama3.2
ollama pull bge-m3
python inference.py
```

## Run with Docker
```bash
docker compose up -d ollama
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull bge-m3
docker compose run --rm app
```
The app container reaches Ollama through the `OLLAMA_URL` environment variable (defaults to `http://localhost:11434` when running without Docker).

## Web app and API

`api.py` serves the assistant as a FastAPI service with a small web page. Each answer comes with the lecture sources it used (video, title, timestamps).

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
# open http://localhost:8000 in a browser
curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "How do I create a link in HTML?"}'
```

With Docker (after pulling the two Ollama models as shown above): `docker compose up -d api`, then open http://localhost:8000. Endpoints: `GET /` (web page), `POST /ask`, `GET /health`, and interactive docs at `/docs`.

## LangChain version

`langchain_pipeline.py` wraps the hybrid retriever as a LangChain retriever and connects it to Ollama (llama3.2) with a LangChain chain (retriever, prompt, LLM, output parser). Run it with `python langchain_pipeline.py` (Ollama must be running).

## Evaluate retrieval quality
```bash
python evaluate_retrieval.py --n 200     # quick check
python evaluate_retrieval.py --n 1000    # full evaluation
```
Generates paraphrased student questions from sampled chunks with llama3.2, then reports Recall@1, Recall@5 and MRR@5 for dense-only, keyword-only, hybrid, and the full reranked pipeline. Results are written to `evaluation_results.md`. It is a strict test: only the exact source chunk counts as a hit.

## Example Query
```
Ask a Question: Where is HTML concluded in this course?

Response: HTML is concluded in Video 13 titled "Entities, Code tag and more on HTML"
at timestamp 520.32 seconds (8 minutes 40 seconds).
```

## Author
Kaveri Anil Ghatage
LinkedIn: kaverighatage | GitHub: Kaveri-Ghatage

## Achievements
- Kaggle Top 10%: Ranked 367/3,724 | XGBoost + SMOTE | 0.89 AUC-ROC
- GATE 2025 Qualified (ECE)
- IEEE Published Researcher 2025
