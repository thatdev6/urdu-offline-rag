"""
embed_index.py — Embed chunks with BGE-M3, build a FAISS index.

Saves both the FAISS index and a sidecar JSON mapping index positions
back to chunk metadata (needed since FAISS only stores vectors, not text).
"""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import CORPUS_CHUNKS, CHUNKS_FILE, INDEX_FILE, METADATA_FILE, EMBED_MODEL_NAME as MODEL_NAME


def load_chunks() -> list[dict]:
    chunks = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def main():
    print(f"Loading chunks from {CHUNKS_FILE}...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading embedding model: {MODEL_NAME} (this may take a while on first run)...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in chunks]
    print("Encoding chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,  # so we can use inner product = cosine similarity
    )
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine sim
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))
    print(f"FAISS index written to {INDEX_FILE} ({index.ntotal} vectors)")

    # Save metadata so we can map FAISS result indices back to chunk text/source
    metadata = [{"id": c["id"], "source": c["source"], "text": c["text"]} for c in chunks]
    METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Metadata written to {METADATA_FILE}")


if __name__ == "__main__":
    main()
