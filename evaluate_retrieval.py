"""
Measures retrieval quality with reproducible numbers. Questions are written by llama3.2 from
random transcript segments (seed 42) and cached in eval_questions.json. Four setups are compared:
dense (FAISS), keyword (BM25), hybrid (RRF) and full (hybrid + cross-encoder rerank).

Run (Ollama running with llama3.2 and bge-m3):
    python evaluate_retrieval.py --n 1000
Merged passages, same questions:
    EMBEDDINGS_FILE=embeddings_merged.joblib INDEX_FILE=faiss_index_merged.bin python evaluate_retrieval.py --n 1000
Strict test: only the passage containing the source segment counts as a hit.
"""
import argparse, json, os, random

import joblib
import numpy as np
import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
QUESTIONS_FILE, ORIGINAL_FILE, K = "eval_questions.json", "embeddings.joblib", 5


def generate_question(text):
    prompt = ("Below is an excerpt from a web development lecture transcript. Write ONE question "
              "a student might ask whose answer is in this excerpt. Paraphrase: do not copy exact "
              "phrases from the excerpt. Return only the question.\n\nExcerpt:\n" + text)
    r = requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": "llama3.2", "prompt": prompt, "stream": False}, timeout=120)
    return r.json()["response"].strip().strip('"')


def build_questions(df, n, seed, min_words=10):
    rows = df[df["text"].str.split().str.len() >= min_words]
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


def segs(c, covers):
    return covers[c] if covers and c in covers else {c}


def score(ranked_lists, truths, k=K, covers=None):
    r1 = r5 = mrr = 0.0
    for ranked, truth in zip(ranked_lists, truths):
        for pos, c in enumerate(list(ranked)[:k]):
            if truth in segs(c, covers):
                r5 += 1
                mrr += 1.0 / (pos + 1)
                r1 += pos == 0
                break
    n = len(truths)
    return {"recall@1": r1 / n, "recall@5": r5 / n, "mrr@5": mrr / n}


def neighbour_map(df):
    """segment id -> {itself, previous, next segment in the same video}."""
    mapping = {}
    for _, g in df.sort_values(["number", "start"]).groupby("number"):
        ids = g["chunk_id"].astype(int).tolist()
        for i, cid in enumerate(ids):
            mapping[cid] = {cid} | set(ids[max(0, i - 1):i]) | set(ids[i + 1:i + 2])
    return mapping


def covered(ranked, covers, k=K):
    got = set()
    for c in list(ranked)[:k]:
        got |= segs(c, covers)
    return got


def score_window(ranked_lists, truths, mapping, k=K, covers=None):
    hit = sum(1 for r, t in zip(ranked_lists, truths) if mapping.get(t, {t}) & covered(r, covers, k))
    return hit / len(truths)


def write_failure_cases(pairs, ranked_lists, orig, passages, covers, mapping,
                        n=8, seed=42, path="failure_cases.md"):
    misses = [(p, r) for p, r in zip(pairs, ranked_lists) if p["chunk_id"] not in covered(r, covers)]
    near = sum(1 for p, r in misses if mapping.get(p["chunk_id"], {p["chunk_id"]}) & covered(r, covers))
    chosen = random.Random(seed).sample(misses, min(n, len(misses)))
    src, got = orig.set_index("chunk_id"), passages.set_index("chunk_id")

    def where(row):
        s = float(row["start"])
        return f"Video {row['number']}: {row['title']} ({int(s // 60)}:{int(s % 60):02d})"

    lines = ["# Failure cases (full pipeline)\n",
             f"{len(misses)} of {len(pairs)} questions missed the source in the top 5; "
             f"{near} of those were near-misses (an adjacent segment was retrieved).\n"]
    for i, (p, r) in enumerate(chosen, 1):
        e, t = src.loc[p["chunk_id"]], got.loc[r[0]]
        lines += [f"## Case {i}", f"**Question:** {p['question']}\n",
                  f"**Expected:** {where(e)}  \n> {str(e['text'])[:300]}\n",
                  f"**Retrieved first:** {where(t)}  \n> {str(t['text'])[:300]}\n",
                  "**Why it failed:** _(read the two passages and write one line here)_\n"]
    open(path, "w").write("\n".join(lines))
    return len(misses), near


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-words", type=int, default=10)
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()

    import faiss
    import hybrid_retrieval as hr

    merged = "source_ids" in hr.df.columns
    covers = ({int(c): set(s) for c, s in zip(hr.df["chunk_id"], hr.df["source_ids"])}
              if merged else None)
    orig = joblib.load(ORIGINAL_FILE) if merged else hr.df

    random.seed(args.seed)
    if os.path.exists(QUESTIONS_FILE) and not args.regenerate:
        pairs = json.load(open(QUESTIONS_FILE))
        print(f"Loaded {len(pairs)} cached questions")
    else:
        print(f"Generating {args.n} questions with llama3.2 ...")
        pairs = build_questions(orig, args.n, args.seed, args.min_words)
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

    setups = {
        "Dense only (FAISS)": dense,
        "Keyword only (BM25)": sparse,
        "Hybrid (FAISS + BM25 + RRF)": lambda q: hr.reciprocal_rank_fusion(dense(q), sparse(q)),
        "Full pipeline (hybrid + rerank)": lambda q: hr.hybrid_retrieve(q, final_k=K)["chunk_id"].tolist(),
    }
    mapping = neighbour_map(orig)
    results, rankings = {}, {}
    for name, fn in setups.items():
        print(f"Evaluating: {name}")
        ranked = [fn(p["question"]) for p in pairs]
        rankings[name] = ranked
        results[name] = score(ranked, truths, covers=covers)
        results[name]["recall@5_neighbour"] = score_window(ranked, truths, mapping, covers=covers)

    lines = [f"Evaluated on {len(pairs)} LLM-generated questions (seed {args.seed}); "
             "strict hit = the source chunk (or the merged passage containing it) is in the top 5; "
             "neighbour hit = the source chunk or the segment right before/after it.\n",
             "| Setup | Recall@1 | Recall@5 | MRR@5 | Recall@5 (incl. neighbours) |", "|---|---|---|---|---|"]
    for name, m in results.items():
        lines.append(f"| {name} | {m['recall@1']:.1%} | {m['recall@5']:.1%} | "
                     f"{m['mrr@5']:.3f} | {m['recall@5_neighbour']:.1%} |")
    table = "\n".join(lines)
    print("\n" + table)
    tag = "_merged" if merged else "_segments"
    open(f"evaluation_results{tag}.md", "w").write(table + "\n")
    json.dump({"n": len(pairs), "seed": args.seed, "results": results},
              open(f"evaluation_results{tag}.json", "w"), indent=1)
    n_miss, n_near = write_failure_cases(pairs, rankings["Full pipeline (hybrid + rerank)"], orig,
                                         hr.df, covers, mapping, seed=args.seed,
                                         path=f"failure_cases{tag}.md")
    print(f"\nFull pipeline missed {n_miss}/{len(pairs)} questions ({n_near} near-misses)")
    print(f"Saved evaluation_results{tag}.md/.json and failure_cases{tag}.md")


if __name__ == "__main__":
    main()
