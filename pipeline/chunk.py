"""
chunk.py — Split clean text files into retrieval-sized chunks.

Uses a word-count-based split (not token count) since we don't want a
tokenizer dependency at this stage. Urdu inflates token counts vs. English,
so chunk sizes here are intentionally conservative.
"""

import json
from pathlib import Path

CORPUS_CLEAN = Path.home() / "Downloads/Projects/manar-prep/corpus/clean"
CORPUS_CHUNKS = Path.home() / "Downloads/Projects/manar-prep/corpus/chunks"

CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 40


def chunk_text(text: str, source: str) -> list[dict]:
    words = text.split()
    chunks = []
    start = 0
    idx = 0

    while start < len(words):
        end = start + CHUNK_SIZE_WORDS
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        if chunk_text_str.strip():
            chunks.append({
                "id": f"{source}_{idx:04d}",
                "source": source,
                "chunk_index": idx,
                "text": chunk_text_str,
            })
            idx += 1

        start += CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS

    return chunks


def main():
    CORPUS_CHUNKS.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(CORPUS_CLEAN.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {CORPUS_CLEAN}")
        return

    all_chunks = []

    for txt_path in txt_files:
        text = txt_path.read_text(encoding="utf-8")
        source = txt_path.stem
        chunks = chunk_text(text, source)
        all_chunks.extend(chunks)
        print(f"{source}: {len(chunks)} chunks")

    out_path = CORPUS_CHUNKS / "chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_chunks)} chunks from {len(txt_files)} files")
    print(f"Written to: {out_path}")


if __name__ == "__main__":
    main()
