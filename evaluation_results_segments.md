Evaluated on 1000 LLM-generated questions (seed 42); strict hit = the source chunk (or the merged passage containing it) is in the top 5; neighbour hit = the source chunk or the segment right before/after it.

| Setup | Recall@1 | Recall@5 | MRR@5 | Recall@5 (incl. neighbours) |
|---|---|---|---|---|
| Dense only (FAISS) | 32.3% | 50.8% | 0.394 | 54.5% |
| Keyword only (BM25) | 12.0% | 26.0% | 0.175 | 29.2% |
| Hybrid (FAISS + BM25 + RRF) | 25.0% | 49.1% | 0.337 | 52.9% |
| Full pipeline (hybrid + rerank) | 39.3% | 57.2% | 0.463 | 59.9% |
