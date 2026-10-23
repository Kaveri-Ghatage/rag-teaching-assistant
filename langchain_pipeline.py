"""
LangChain version of the teaching assistant.

Wraps the hybrid retriever (FAISS + BM25 + RRF + cross-encoder rerank) as a LangChain
Retriever and connects it to a local Ollama LLM with a LangChain (LCEL) chain.

Run (Ollama must be running with llama3.2 and bge-m3 pulled):
    python langchain_pipeline.py
"""
import os
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

PROMPT = ChatPromptTemplate.from_template(
    """You are a teaching assistant for a web development course.
Below are lecture subtitle chunks (video title, video number, start and end time in seconds, text).

{context}
---------------------------------
Question: {question}

Answer in a human way. Say in which video and at what timestamp the topic is taught and guide
the student to that video. Do not mention the chunk format. If the question is unrelated to the
course, say you can only answer questions about the course."""
)


class HybridRetriever(BaseRetriever):
    """LangChain retriever backed by hybrid_retrieve() from hybrid_retrieval.py."""

    final_k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        from hybrid_retrieval import hybrid_retrieve  # lazy: loads index on first use

        df = hybrid_retrieve(query, dense_k=20, sparse_k=20, fused_k=15, final_k=self.final_k)
        return [
            Document(
                page_content=row["text"],
                metadata={
                    "title": row["title"],
                    "number": row["number"],
                    "start": row["start"],
                    "end": row["end"],
                },
            )
            for _, row in df.iterrows()
        ]


def format_docs(docs: List[Document]) -> str:
    return "\n".join(
        f'[Video {d.metadata["number"]}: {d.metadata["title"]} | '
        f'{d.metadata["start"]}s-{d.metadata["end"]}s] {d.page_content}'
        for d in docs
    )


def build_chain(llm=None, retriever=None):
    """Build retriever -> prompt -> LLM -> text chain. Pass llm/retriever to override."""
    if llm is None:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model="llama3.2", base_url=OLLAMA_URL)
    retriever = retriever or HybridRetriever()
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


if __name__ == "__main__":
    chain = build_chain()
    print(chain.invoke(input("Ask a Question: ")))
