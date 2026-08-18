"""
rag.py — Query -> retrieve (SIJILL) -> prompt -> generate, with citations.

Requires llama-server running separately:
  ./build/bin/llama-server -m <model.gguf> -c 4096 --port 8080
"""

import json

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from config import INDEX_FILE, METADATA_FILE, EMBED_MODEL_NAME, LLAMA_SERVER_URL, TOP_K, REQUEST_TIMEOUT

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about university "
    "policies using ONLY the provided document excerpts. "
    "Answer in the same language/script the question was asked in "
    "(Roman Urdu, Urdu script, or English). "
    "Keep answers short and plain — no markdown headers, no emoji, no bullet "
    "styling. If the answer is not in the provided context, say clearly that "
    "you don't have that information rather than guessing. "
    "Always mention which document your answer is based on."
)


class RAGPipeline:
    def __init__(self):
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        print("Loading FAISS index...")
        self.index = faiss.read_index(str(INDEX_FILE))
        self.metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        print(f"Ready — {self.index.ntotal} chunks indexed.")

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        query_vec = self.embed_model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype="float32")
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.metadata[idx]
            results.append({**chunk, "score": float(score)})
        return results

    def build_prompt(self, query: str, chunks: list[dict]) -> str:
        context_parts = []
        for c in chunks:
            context_parts.append(f"[Source: {c['source']}]\n{c['text']}")
        context = "\n\n---\n\n".join(context_parts)
        return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    def generate(self, prompt: str) -> str:
        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 200,
        }
        resp = requests.post(LLAMA_SERVER_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def answer(self, query: str, k: int = TOP_K, verbose: bool = True) -> dict:
        chunks = self.retrieve(query, k)
        prompt = self.build_prompt(query, chunks)

        if verbose:
            print(f"\nRetrieved {len(chunks)} chunks:")
            for c in chunks:
                print(f"  [{c['score']:.3f}] {c['source']} — {c['text'][:80]}...")

        answer_text = self.generate(prompt)
        return {
            "query": query,
            "answer": answer_text,
            "sources": [c["source"] for c in chunks],
            "retrieved_chunks": chunks,
        }


def main():
    rag = RAGPipeline()
    print("\nType a question (Roman Urdu, Urdu script, or English). Ctrl+C to exit.\n")
    while True:
        try:
            query = input("> ").strip()
            if not query:
                continue
            result = rag.answer(query)
            print(f"\n{result['answer']}\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
