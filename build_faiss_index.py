"""
Step 4: build the FAISS IVF-Flat index (faiss_index.bin) from embeddings.joblib.

Matches the index hybrid_retrieval.py expects: inner-product metric on
L2-normalized vectors, 100 clusters (nlist), wrapped in an ID map so each
vector keeps its chunk_id, and nprobe=10 at search time.
"""
import joblib
import numpy as np
import faiss

NLIST = 100   # number of IVF clusters
NPROBE = 10   # same value hybrid_retrieval.py sets at load time


def build_index(vectors, ids):
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    faiss.normalize_L2(vectors)  # hybrid_retrieval.py also normalizes the query
    dim = vectors.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, NLIST, faiss.METRIC_INNER_PRODUCT)
    index = faiss.IndexIDMap(ivf)
    index.train(vectors)
    index.add_with_ids(vectors, np.asarray(ids, dtype="int64"))
    ivf.nprobe = NPROBE
    return index


if __name__ == "__main__":
    df = joblib.load("embeddings.joblib")
    vectors = np.vstack(df["embedding"].values)
    index = build_index(vectors, df["chunk_id"].values)
    faiss.write_index(index, "faiss_index.bin")
    print(f"Indexed {index.ntotal} vectors (dim {vectors.shape[1]}) into faiss_index.bin")
