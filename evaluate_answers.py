"""
Checks the quality of the generated ANSWERS (retrieval quality is measured by evaluate_retrieval.py).

Three checks, each reported with its sample size:
1. Grounding: for N in-course questions (from eval_questions.json), the answer is judged against the
   retrieved excerpts: "is every claim in the answer supported by the excerpts?" (YES/NO)
2. Citation: the answer names a video number that really is among the retrieved sources (a plain
   text check, no language model involved).
3. Refusal: for out-of-course questions, the answer should say the course does not cover it
   (judged YES/NO).

Important limit: the judge is the same small local model (llama3.2), so checks 1 and 3 are
indicative, not a precise score. Check 2 is exact. The script also prints a few examples so the
results can be inspected by hand.

Run (Ollama running with llama3.2 and bge-m3; use the merged index):
    EMBEDDINGS_FILE=embeddings_merged.joblib INDEX_FILE=faiss_index_merged.bin python evaluate_answers.py --n 50
Needs eval_questions.json (created by evaluate_retrieval.py). Writes answer_quality.md.
"""
import argparse
import json
import random
import re
import statistics
import time

import requests

import api  # reuses the exact retrieval, prompt and LLM call the web app uses

OUT_OF_COURSE = [
    "Who won the cricket world cup?",
    "What is the capital of France?",
    "How do I cook chicken biryani?",
    "What is the current stock price of Tesla?",
    "Explain how quantum computers work.",
    "How do I train a neural network in PyTorch?",
    "What is the best phone to buy this year?",
    "How do I lose weight fast?",
    "Who is the prime minister of India?",
    "Write me a poem about the ocean.",
    "How do I file income tax returns?",
    "What is the plot of the movie Inception?",
    "How does a car engine work?",
    "Explain the French revolution.",
    "How do I fix my washing machine?",
]

def judge(prompt):
    """Ask the local model a YES/NO question; return True for YES."""
    r = requests.post(f"{api.OLLAMA_URL}/api/generate",
                      json={"model": api.LLM_MODEL, "prompt": prompt, "stream": False,
                            "options": {"temperature": 0}}, timeout=120)
    return r.json()["response"].strip().upper().startswith("YES")


def excerpts_text(df):
    return "\n".join(f"- (video {r['number']}) {r['text']}" for _, r in df.iterrows())


def answer_question(question):
    start = time.time()
    df = api.retrieve(question, 5)
    if not api.is_covered(df):
        return df, api.NOT_COVERED, time.time() - start
    answer = api.ask_llm(api.build_prompt(question, df))
    return df, answer, time.time() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of in-course questions")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = json.load(open("eval_questions.json"))
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.n]

    grounded = cited = 0
    latencies, examples = [], []
    for i, p in enumerate(pairs, 1):
        df, answer, sec = answer_question(p["question"])
        latencies.append(sec)
        ok = judge(
            "EXCERPTS:\n" + excerpts_text(df) + "\n\nANSWER:\n" + answer +
            "\n\nIs every factual claim in the ANSWER supported by the EXCERPTS? "
            "Reply with only YES or NO.")
        grounded += ok
        shown = {str(n) for n in df["number"]}
        said = set(re.findall(r"video\s*#?\s*(\d+)", answer.lower()))
        cited += bool(said & shown)
        if len(examples) < 5:
            examples.append((p["question"], answer, ok))
        if i % 10 == 0:
            print(f"  answered {i}/{len(pairs)}")
              refused = 0
    out_rows = []
    for q in OUT_OF_COURSE:
        _, answer, _ = answer_question(q)
        ok = judge(
            "QUESTION:\n" + q + "\n\nANSWER:\n" + answer +
            "\n\nDoes the ANSWER say that the course does not cover this question (or decline to "
            "answer it), instead of answering it? Reply with only YES or NO.")
        refused += ok
        out_rows.append((q, answer, ok))

    n, m = len(pairs), len(OUT_OF_COURSE)
    lines = [
        "# Answer quality",
        "",
        f"In-course questions: {n} sampled from the evaluation set (seed {args.seed}). "
        f"Out-of-course questions: {m} written by hand.",
        "",
        "| Check | Result | How it is measured |",
        "|---|---|---|",
        f"| Answer supported by retrieved excerpts | {grounded / n:.0%} ({grounded}/{n}) | "
        "llama3.2 as judge (indicative) |",
        f"| Answer cites a video that was retrieved | {cited / n:.0%} ({cited}/{n}) | "
        "text check (exact) |",
        f"| Refuses out-of-course questions | {refused / m:.0%} ({refused}/{m}) | "
        "llama3.2 as judge (indicative) |",
        f"| Median time per answer (retrieval + generation) | {statistics.median(latencies):.1f} s | "
        "measured |",
        "",
        "## In-course examples",
    ]
    for q, a, ok in examples:
        lines += [f"- **Q:** {q}  ", f"  **A:** {a.strip()}  ", f"  Judged supported: {'yes' if ok else 'no'}"]
    lines += ["", "## Out-of-course examples (first 5)"]
    for q, a, ok in out_rows[:5]:
        lines += [f"- **Q:** {q}  ", f"  **A:** {a.strip()}  ", f"  Judged as refusal: {'yes' if ok else 'no'}"]
    open("answer_quality.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    print("\nSaved answer_quality.md")


if __name__ == "__main__":
    main()
