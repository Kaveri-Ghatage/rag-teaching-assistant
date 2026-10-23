"""
Hybrid retrieval pipeline: FAISS dense search + BM25 keyword search,
fused with Reciprocal Rank Fusion, then refined with a cross-encoder reranker.

(Kaveri Anil Ghatage - RAG Teaching Assistant)

WHY HYBRID RETRIEVAL, so you can explain this in an interview:
- FAISS (dense/semantic search) is great at understanding MEANING. It will
  match "how do I make text bold" to a chunk that says "use the <strong> tag"
  even with zero overlapping words, because the embeddings capture meaning.
- BM25 (sparse/keyword search) is great at EXACT terms. If a student asks
  "what is z-index", BM25 reliably surfaces chunks that literally contain
  "z-index" - something dense search can sometimes under-rank if the
  embedding model wasn't trained heavily on that exact jargon.
- Using both and fusing the results covers each method's blind spot.

WHY RECIPROCAL RANK FUSION (RRF), not just averaging scores:
FAISS returns inner-product scores; BM25 returns TF-IDF-based scores. These
two scales are NOT comparable (e.g. FAISS might range 0-1, BM25 might range
0-20) so averaging them directly is meaningless without careful tuning.
RRF sidesteps this entirely: it only looks at each chunk's RANK in each
list (1st place, 2nd place, ...), not the raw score. A chunk ranked #1 in
both lists scores higher than one ranked #1 in only one list. This is the
same fusion technique used in real hybrid search systems (e.g. Elasticsearch
8.8+'s RRF retriever).

WHY A CROSS-ENCODER ON TOP OF THAT:
FAISS and BM25 are both "fast but approximate" - they embed/score the query
and each document SEPARATELY, then compare. A cross-encoder instead reads
the query and a candidate chunk TOGETHER in one forward pass, so it can
catch relationships a separate-embedding approach misses. It's too slow to
run against all 7,252 chunks, so we only rerank the ~15 candidates RRF
already shortlisted - this is the standard "retrieve cheap, rerank precise"
pattern used in production search systems.
"""
import joblib
import numpy as np
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from embedding_utils import embed_query

print("Loading real chunk data and indices...")
df = joblib.load("embeddings.joblib")
faiss_index = faiss.read_index("faiss_index.bin")
faiss.extract_index_ivf(faiss_index).nprobe = 10

# Build BM25 index once at startup (cheap - milliseconds for 7,252 chunks)
tokenized_corpus = [doc.lower().split() for doc in df["text"].tolist()]
bm25 = BM25Okapi(tokenized_corpus)
chunk_ids = df["chunk_id"].tolist()

# Cross-encoder is loaded lazily on first actual use (not at import time).
# First call downloads the model (~90MB) from Hugging Face - needs internet
# once, then it's cached locally for every run after that.
_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def reciprocal_rank_fusion(dense_ids, sparse_ids, k=60):
    """
    dense_ids / sparse_ids: ranked lists of chunk_ids (best match first).
    k=60 is the standard RRF damping constant from the original paper
    (Cormack et al., 2009) - it just softens the impact of rank differences
    so a #1 vs #2 ranking doesn't swing the score wildly.
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
