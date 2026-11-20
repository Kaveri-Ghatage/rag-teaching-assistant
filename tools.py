from hybrid_retrieval import hybrid_retrieve

# Below this reranker score, the question is treated as out of course
REFUSE_THRESHOLD = -7.44


def search_lectures(query):
    df = hybrid_retrieve(query, dense_k=20, sparse_k=20, fused_k=15, final_k=5)
    best_score = float(df["rerank_score"].max())

    if best_score < REFUSE_THRESHOLD:
        return {"in_course": False, "chunks": []}

    chunks = df[["title", "number", "start", "end", "text"]].to_dict(orient="records")
    return {"in_course": True, "chunks": chunks}


if __name__ == "__main__":
    print(search_lectures("What is a CSS selector?"))
