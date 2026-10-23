"""
Measures retrieval quality of the RAG teaching assistant with real, reproducible numbers.

How it works
1. Sample chunks from embeddings.joblib (fixed seed).
2. Ask llama3.2 (via Ollama) to write one student question per chunk, paraphrased
   so it does not copy the transcript wording.
3. For each question, check whether the source chunk is retrieved, for four setups:
   dense only (FAISS), keyword only (BM25), hybrid (FAISS + BM25 + RRF), and the full
   pipeline (hybrid + cross-encoder rerank).
4. Report Recall@1, Recall@5 and MRR@5 for each setup.

Run (Ollama must be running with llama3.2 and bge-m3 pulled):
    python evaluate_retrieval.py --n 200      # quick check
    python evaluate_retrieval.py --n 1000     # full evaluation (takes longer)

Questions are cached in eval_questions.json, so re-runs are fast and repeatable.
Note: this is a strict test. Only the exact source chunk counts as a hit, even though
neighbouring chunks may also answer the question, so real-world relevance is likely higher.
"""
import argparse
import json
import os
import random

import numpy as np
import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
QUESTIONS_FILE = "eval_questions.json"
K = 5


def generate_question(text):
    prompt = (
        "Below is an excerpt from a web development lecture transcript. Write ONE question "
        "a student might ask whose answer is in this excerpt. Paraphrase: do not copy exact "
        "phrases from the excerpt. Return only the question.\n\nExcerpt:\n" + text
    )
    r = requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": "llama3.2", "prompt": prompt, "stream": False}, timeout=120)
    return r.json()["response"].strip().strip('"')


def build_questions(df, n, seed):
    rows = df[df["text"].str.split().str.len() >= 20]
    sample = rows.sample(n=min(n, len(rows)), random_state=seed)
    pairs = []
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        try:
            pairs.append({"chunk_id": int(row["chunk_id"]), "question": generate_question(row["text"])})
        except Exception as e:
            print(f"  skipped a chunk ({e})")
        if i % 25 == 0:
            print(f"  generated {i}/{len(sample)} questions")
    return pairs


def score(ranked_lists, truths, k=K):
    """ranked_lists: list of ranked chunk_id lists; truths: the correct chunk_id for each."""
    r1 = r5 = mrr = 0.0
    for ranked, truth in zip(ranked_lists, truths):
        top = list(ranked)[:k]
        if truth in top:
            r5 += 1
            mrr += 1.0 / (top.index(truth) + 1)
            if top[0] == truth:
                r1 += 1
    n = len(truths)
    return {"recall@1": r1 / n, "recall@5": r5 / n, "mrr@5": mrr / n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="number of evaluation questions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regenerate", action="store_true", help="ignore cached questions")
    args = parser.parse_args()

    import faiss  # imported here so score() can be tested without the heavy stack
    import hybrid_retrieval as hr

    random.seed(args.seed)
    if os.path.exists(QUESTIONS_FILE) and not args.regenerate:
        pairs = json.load(open(QUESTIONS_FILE))
        print(f"Loaded {len(pairs)} cached questions from {QUESTIONS_FILE}")
    else:
        print(f"Generating {args.n} questions with llama3.2 ...")
        pairs = build_questions(hr.df, args.n, args.seed)
        json.dump(pairs, open(QUESTIONS_FILE, "w"), indent=1)
    pairs = pairs[: args.n]
    truths = [p["chunk_id"] for p in pairs]

    def dense(q):
        emb = np.array(hr.embed_query(q), dtype="float32").reshape(1, -1)
        faiss.normalize_L2(emb)
        _, ids = hr.faiss_index.search(emb, 20)
        return [int(i) for i in ids[0] if i != -1]

    def sparse(q):
        s = hr.bm25.get_scores(q.lower().split())
        return [hr.chunk_ids[i] for i in np.argsort(s)[::-1][:20]]

    def hybrid(q):
        return hr.reciprocal_rank_fusion(dense(q), sparse(q))

    def full(q):
        return hr.hybrid_retrieve(q, final_k=K)["chunk_id"].tolist()

    setups = {"Dense only (FAISS)": dense, "Keyword only (BM25)": sparse,
              "Hybrid (FAISS + BM25 + RRF)": hybrid, "Full pipeline (hybrid + rerank)": full}
    results = {}
    for name, fn in setups.items():
        print(f"Evaluating: {name}")
        results[name] = score([fn(p["question"]) for p in pairs], truths)

    lines = [f"Evaluated on {len(pairs)} LLM-generated questions (seed {args.seed}); "
             "a hit means the source chunk is in the top 5.\n",
             "| Setup | Recall@1 | Recall@5 | MRR@5 |", "|---|---|---|---|"]
    for name, m in results.items():
        lines.append(f"| {name} | {m['recall@1']:.1%} | {m['recall@5']:.1%} | {m['mrr@5']:.3f} |")
    table = "\n".join(lines)
    print("\n" + table)
    open("evaluation_results.md", "w").write(table + "\n")
    json.dump({"n": len(pairs), "seed": args.seed, "results": results},
              open("evaluation_results.json", "w"), indent=1)
    print("\nSaved evaluation_results.md and evaluation_results.json")


if __name__ == "__main__":
    main()
