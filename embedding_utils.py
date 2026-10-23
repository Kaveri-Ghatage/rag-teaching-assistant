"""Embedding helpers: batched requests to Ollama (bge-m3) and an LRU cache for query embeddings."""
import os
from functools import lru_cache

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
BATCH_SIZE = 32


def _embed_batch(texts):
    r = requests.post(f"{OLLAMA_URL}/api/embed", json={"model": "bge-m3", "input": texts})
    return r.json()["embeddings"]


def create_embedding(text_list, batch_size=BATCH_SIZE):
    """Embed many texts, sending them to Ollama in batches instead of one request per text."""
    embeddings = []
    for start in range(0, len(text_list), batch_size):
        embeddings.extend(_embed_batch(text_list[start:start + batch_size]))
    return embeddings


@lru_cache(maxsize=1024)
def _cached_query_embedding(query):
    return tuple(_embed_batch([query])[0])


def embed_query(query):
    """Embed a single query; repeated queries are served from an LRU cache (no new request)."""
    return list(_cached_query_embedding(query))
