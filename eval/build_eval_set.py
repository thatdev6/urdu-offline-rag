"""
build_eval_set.py — Interactively turn raw queries into a gold eval set.

For each query: retrieves top-12 candidate chunks via SIJILL (BGE-M3 + FAISS),
shows them to you, and lets you mark which are relevant and type gold facts.
Appends completed entries to eval_set.jsonl so you can stop/resume anytime.
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import CORPUS_CHUNKS as CHUNKS_DIR, EVAL_DIR, QUERIES_RAW_FILE as QUERIES_FILE, EVAL_SET_FILE, TOP_K


def load_done_queries() -> set[str]:
    """So you can stop and resume without redoing finished queries."""
    if not EVAL_SET_FILE.exists():
        return set()
    done = set()
    with EVAL_SET_FILE.open() as f:
        for line in f:
            done.add(json.loads(line)["query"])
    return done


def main():
    print("Loading embedding model and index...")
    model = SentenceTransformer("BAAI/bge-m3")
    index = faiss.read_index(str(CHUNKS_DIR / "faiss.index"))
    metadata = json.loads((CHUNKS_DIR / "chunk_metadata.json").read_text())

    queries = [q.strip() for q in QUERIES_FILE.read_text(encoding="utf-8").splitlines() if q.strip()]
    done = load_done_queries()

    print(f"\n{len(queries)} total queries, {len(done)} already done.\n")

    for query in queries:
        if query in done:
            continue

        print("=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        qvec = model.encode([query], normalize_embeddings=True)
        qvec = np.array(qvec, dtype="float32")
        scores, indices = index.search(qvec, TOP_K)

        candidates = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            chunk = metadata[idx]
            candidates.append(chunk)
            print(f"\n[{rank}] score={score:.3f} source={chunk['source']} id={chunk['id']}")
            print(f"    {chunk['text'][:300]}")

        print("\n" + "-" * 80)
        print("Which candidate number(s) actually answer this? (comma-separated, or 'none')")
        selection = input("> ").strip()

        if selection.lower() == "none":
            entry = {
                "query": query,
                "answerable": False,
                "gold_facts": [],
                "source_chunk_ids": [],
                "note": "Not covered in corpus (or not found in top-12)",
            }
        else:
            picked_ranks = [int(x.strip()) for x in selection.split(",")]
            picked_chunks = [candidates[r - 1] for r in picked_ranks]
            source_ids = [c["id"] for c in picked_chunks]

            print("\nType 2-4 short gold facts this answer must contain (one per line, blank line to finish):")
            facts = []
            while True:
                fact = input("  fact> ").strip()
                if not fact:
                    break
                facts.append(fact)

            entry = {
                "query": query,
                "answerable": True,
                "gold_facts": facts,
                "source_chunk_ids": source_ids,
                "note": "",
            }

        with EVAL_SET_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"\nSaved. ({len(done) + 1}/{len(queries)} done)\n")
        done.add(query)

    print("\nAll queries processed.")
    print(f"Eval set written to: {EVAL_SET_FILE}")


if __name__ == "__main__":
    main()
