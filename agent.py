import json
import re
import ollama
from tools import search_lectures

MODEL = "llama3.2"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_lectures",
        "description": "Search the Sigma Web Development course lectures. Use this for any question about web development topics.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
}]

SYSTEM = (
    "You are a teaching assistant for the Sigma Web Development course. "
    "If the user only greets you or says thanks, reply briefly and politely WITHOUT calling any tool. "
    "For every other question, even if it seems unrelated to web development or you think you know the answer, "
    "you MUST call the search_lectures tool first and never answer from your own knowledge. "
    "Then answer in a friendly way using ONLY the lecture chunks returned. "
    "Say which video number and at what timestamp (in seconds) the topic is taught, "
    "and guide the student to that video. Do not make up information."
)

REFUSAL = "Sorry, I can only answer questions related to the course."

# A message is small talk only if EVERY word is in this list
SMALL_TALK_WORDS = {
    "hi", "hii", "hello", "hey", "thanks", "thank", "you", "that", "helped",
    "so", "much", "a", "lot", "ok", "okay", "bye", "goodbye", "good",
    "morning", "afternoon", "evening", "great", "awesome", "cool", "nice",
}


def is_small_talk(text):
    words = re.findall(r"[a-z']+", text.lower())
    return len(words) > 0 and all(w in SMALL_TALK_WORDS for w in words)


def run_agent(question):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    if is_small_talk(question):
        return ollama.chat(model=MODEL, messages=messages)["message"]["content"]

    response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
    msg = response["message"]
    tool_calls = msg.get("tool_calls") or []

    if not tool_calls:
        return msg["content"]

    messages.append(msg)
    for call in tool_calls:
        args = call["function"]["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        query = args.get("query", question)
        print(f"[tool call] search_lectures(query={query!r})")
        result = search_lectures(query)
        if not result["in_course"]:
            print("[agent] refused: question is out of course")
            return REFUSAL
        messages.append({
            "role": "tool",
            "name": "search_lectures",
            "content": json.dumps(result, default=str),
        })

    final = ollama.chat(model=MODEL, messages=messages)
    return final["message"]["content"]


if __name__ == "__main__":
    while True:
        q = input("Ask a question (or type exit): ").strip()
        if q.lower() == "exit":
            break
        if q:
            print("\n" + run_agent(q) + "\n")