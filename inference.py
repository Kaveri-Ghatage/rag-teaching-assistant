import os
import requests
from hybrid_retrieval import hybrid_retrieve

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def inference(prompt):
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    print(response)
    return response


incoming_query = input("Ask a Question: ")

# Hybrid retrieval: FAISS dense search + BM25 keyword search, fused with
# Reciprocal Rank Fusion, then refined by a cross-encoder reranker.
# See hybrid_retrieval.py for the full explanation of why each stage exists.
new_df = hybrid_retrieve(incoming_query, dense_k=20, sparse_k=20, fused_k=15, final_k=5)
# print(new_df[["title", "number", "text", "rerank_score"]])

prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''
with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)
# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])
