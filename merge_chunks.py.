"""
Merge the short Whisper segments into larger, overlapping passages and rebuild the search index.

Why: the original chunks are single transcript segments (median about 6 words). That is too little
text for embeddings and rerankers to work with, and evaluation showed it is the main limit on
retrieval quality. This script groups consecutive segments of the same video into passages of
about --target-words words, with --overlap-words words shared between neighbouring passages.

Run (Ollama must be running with bge-m3 pulled):
    python merge_chunks.py
Outputs: embeddings_merged.joblib and faiss_index_merged.bin. Use them with:
    EMBEDDINGS_FILE=embeddings_merged.joblib INDEX_FILE=faiss_index_merged.bin python evaluate_retrieval.py --n 1000
Each merged passage keeps `source_ids` (the original segment ids it covers), so evaluation can
reuse the same questions and count a hit when the retrieved passage contains the source segment.
"""
import argparse

import joblib
import numpy as np
import pandas as pd

TARGET_WORDS = 60
OVERLAP_WORDS = 20


def merge_segments(df, target_words=TARGET_WORDS, overlap_words=OVERLAP_WORDS):
    """Return a DataFrame of merged passages (no embeddings yet), with new chunk_ids from 0."""
    rows = []
    for number, g in df.sort_values(["number", "start"]).groupby("number", sort=True):
        segs = g.to_dict("records")
        words = [len(str(s["text"]).split()) for s in segs]
        i = 0
        while i < len(segs):
            j, total = i, 0
            while j < len(segs) and total < target_words:
                total += words[j]
                j += 1
            part = segs[i:j]
            rows.append({
                "number": number,
                "title": part[0]["title"],
                "start": part[0]["start"],
                "end": part[-1]["end"],
                "text": " ".join(str(s["text"]).strip() for s in part),
                "source_ids": [int(s["chunk_id"]) for s in part],
            })
            if j >= len(segs):
                break
            k, shared = j, 0           # step back to share about overlap_words words
            while k > i + 1 and shared < overlap_words:
                k -= 1
                shared += words[k]
            i = max(k, i + 1)
    out = pd.DataFrame(rows)
    out["chunk_id"] = range(len(out))
    return out
  
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-words", type=int, default=TARGET_WORDS)
    parser.add_argument("--overlap-words", type=int, default=OVERLAP_WORDS)
    args = parser.parse_args()

    from embedding_utils import create_embedding
    import faiss

    df = joblib.load("embeddings.joblib")
    merged = merge_segments(df, args.target_words, args.overlap_words)
    print(f"{len(df)} segments -> {len(merged)} passages "
          f"(median {int(merged['text'].str.split().str.len().median())} words)")

    merged["embedding"] = create_embedding(merged["text"].tolist())
    joblib.dump(merged, "embeddings_merged.joblib")

    vectors = np.ascontiguousarray(np.vstack(merged["embedding"].values), dtype="float32")
    faiss.normalize_L2(vectors)                            # same as hybrid_retrieval.py does for queries
    nlist = max(1, int(len(merged) ** 0.5))                # fewer vectors need fewer clusters
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(vectors.shape[1]), vectors.shape[1], nlist,
                             faiss.METRIC_INNER_PRODUCT)
    ivf.train(vectors)
    index = faiss.IndexIDMap(ivf)
    index.add_with_ids(vectors, merged["chunk_id"].values.astype("int64"))
    faiss.write_index(index, "faiss_index_merged.bin")
    print(f"Saved embeddings_merged.joblib and faiss_index_merged.bin ({index.ntotal} vectors, nlist={nlist})")


if __name__ == "__main__":
    main()
