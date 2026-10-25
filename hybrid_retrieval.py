"""
Hybrid retrieval pipeline: FAISS dense search + BM25 keyword search,
fused with Reciprocal Rank Fusion, then refined with a cross-encoder reranker.
(Kaveri Anil Ghatage - RAG Teaching Assistant)

Why hybrid: dense search (FAISS) matches MEANING, keyword search (BM25) matches EXACT terms
such as "z-index"; using both covers each method's blind spot.
Why RRF: FAISS and BM25 scores are on different scales, so RRF fuses by RANK, not by score.
Why a cross-encoder: it reads the query and a candidate passage TOGETHER, which is more precise
than comparing separate embeddings, but too slow for every passage, so it only reranks the
~15 candidates RRF shortlisted ("retrieve cheap, rerank precise").
"""
import os

import joblib
import numpy as np
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from embedding_utils import embed_query

print("Loading real chunk data and indices...")
# Defaults use the original per-segment data; set EMBEDDINGS_FILE / INDEX_FILE to use the merged passages
# (embeddings_merged.joblib / faiss_index_merged.bin).
df = joblib.load(os.environ.get("EMBEDDINGS_FILE", "embeddings.joblib"))
faiss_index = faiss.read_index(os.environ.get("INDEX_FILE", "faiss_index.bin"))
faiss.extract_index_ivf(faiss_index).nprobe = 10

# Build BM25 index once at startup (cheap - milliseconds)
tokenized_corpus = [doc.lower().split() for doc in df["text"].tolist()]
bm25 = BM25Okapi(tokenized_corpus)
chunk_ids = df["chunk_id"].tolist()

# Cross-encoder is loaded lazily on first use. The first call downloads the model (~90MB)
# from Hugging Face; after that it is cached locally.
_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def reciprocal_rank_fusion(dense_ids, sparse_ids, k=60):
    """
    dense_ids / sparse_ids: ranked lists of chunk_ids (best match first).
    k=60 is the standard RRF damping constant (Cormack et al., 2009).
    Returns chunk_ids sorted by fused score, best first.
    """
    scores = {}
    for rank, cid in enumerate(dense_ids):
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
    for rank, cid in enumerate(sparse_ids):
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda c: scores[c], reverse=True)


def hybrid_retrieve(query, dense_k=20, sparse_k=20, fused_k=15, final_k=5):
    # --- Dense retrieval (FAISS) ---
    q_emb = np.array(embed_query(query), dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q_emb)
    _, result_ids = faiss_index.search(q_emb, dense_k)
    dense_ids = [int(i) for i in result_ids[0] if i != -1]

    # --- Sparse retrieval (BM25) ---
    bm25_scores = bm25.get_scores(query.lower().split())
    sparse_order = np.argsort(bm25_scores)[::-1][:sparse_k]
    sparse_ids = [chunk_ids[i] for i in sparse_order]

    # --- Fuse with RRF ---
    fused_ids = reciprocal_rank_fusion(dense_ids, sparse_ids)[:fused_k]
    candidates = df[df["chunk_id"].isin(fused_ids)].set_index("chunk_id").loc[fused_ids].reset_index()

    # --- Cross-encoder rerank of the shortlisted candidates ---
    pairs = [[query, row["text"]] for _, row in candidates.iterrows()]
    rerank_scores = get_reranker().predict(pairs)
    candidates["rerank_score"] = rerank_scores
    final = candidates.sort_values("rerank_score", ascending=False).head(final_k)
    return final.reset_index(drop=True)


if __name__ == "__main__":
    query = input("Ask a Question: ")
    results = hybrid_retrieve(query)
    print(results[["chunk_id", "number", "title", "text", "rerank_score"]].to_string(index=False))
