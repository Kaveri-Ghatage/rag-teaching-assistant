Evaluated on 1000 LLM-generated questions (seed 42); strict hit = the source chunk (or the merged passage containing it) is in the top 5; neighbour hit = the source chunk or the segment right before/after it.

| Setup | Recall@1 | Recall@5 | MRR@5 | Recall@5 (incl. neighbours) |
|---|---|---|---|---|
| Dense only (FAISS) | 33.0% | 54.7% | 0.411 | 56.1% |
| Keyword only (BM25) | 22.7% | 38.8% | 0.281 | 40.4% |
| Hybrid (FAISS + BM25 + RRF) | 31.7% | 55.7% | 0.401 | 57.1% |
| Full pipeline (hybrid + rerank) | 40.9% | 64.0% | 0.495 | 64.8% |
