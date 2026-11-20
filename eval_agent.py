# eval_agent.py - runs eval_set.py questions through the agent's routing step
import csv
import ollama
from agent import MODEL, TOOLS, SYSTEM, is_small_talk
from tools import search_lectures
from eval_set import EVAL_SET

rows = []
for item in EVAL_SET:
    q, kind = item["q"], item["kind"]
    if is_small_talk(q):
        called = False
    else:
        resp = ollama.chat(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": q}],
            tools=TOOLS,
        )
        calls = resp["message"].get("tool_calls") or []
        called = len(calls) > 0
    in_course, videos = "", ""
    if called:
        result = search_lectures(q)
        in_course = result["in_course"]
        videos = ",".join(str(c["number"]) for c in result["chunks"])

    if kind == "no_tool":
        passed = not called
    elif kind == "in_course":
        passed = called and in_course is True
    else:  # out_of_course
        passed = called and in_course is False

    rows.append({"question": q, "kind": kind, "tool_called": called,
                 "in_course": in_course, "videos_top5": videos, "pass": passed})
    print(("PASS" if passed else "FAIL"), "|", kind, "|", q)

with open("eval_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

for k in ["in_course", "out_of_course", "no_tool"]:
    sub = [r for r in rows if r["kind"] == k]
    ok = sum(r["pass"] for r in sub)
    print(f"{k}: {ok}/{len(sub)}")
print(f"overall: {sum(r['pass'] for r in rows)}/{len(rows)}")