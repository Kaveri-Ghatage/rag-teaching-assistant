import json
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
    "For any question about web development, call the search_lectures tool first. "
    "Then answer in a friendly way using ONLY the lecture chunks returned. "
    "Say which video number and at what timestamp (in seconds) the topic is taught, "
    "and guide the student to that video. Do not make up information."
)


def run_agent(question):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    resp = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
    msg = resp.message

    if not msg.tool_calls:
        print("[agent] no tool was called")
        return msg.content

    messages.append(msg)
    for call in msg.tool_calls:
        args = call.function.arguments
        print(f"[tool call] {call.function.name}({args})")
        query = args.get("query", question)
        result = search_lectures(query)

        if not result["in_course"]:
            print("[agent] refused: question is out of course")
            return "Sorry, I can only answer questions related to the course."

        messages.append({
            "role": "tool",
            "tool_name": call.function.name,
            "content": json.dumps(result["chunks"], default=str),
        })

    final = ollama.chat(model=MODEL, messages=messages)
    return final.message.content


if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type exit): ")
        if q.strip().lower() == "exit":
            break
        print("\n" + run_agent(q))