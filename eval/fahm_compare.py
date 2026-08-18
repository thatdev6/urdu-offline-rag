"""
fahm_compare.py — Compare retrieval rank: raw vs. lexically-normalized query.

For each answerable eval query, checks what rank the known-correct chunk(s)
appear at, using raw vs. FAHM-Arm-1-normalized query text. Retrieval-only,
no LLM calls -- fast, no memory/timeout risk.
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from fahm_normalize import normalize_roman_urdu  # noqa: E402

BASE = Path.home() / "Downloads/Projects/manar-prep"
CHUNKS_DIR = BASE / "corpus/chunks"
EVAL_SET_FILE = BASE / "eval/eval_set.jsonl"
RESULTS_FILE = BASE / "eval/results/fahm_comparison.json"

SEARCH_K = 20  # look deep enough to find rank even if it's not in a small top-k


def find_rank(index, metadata, query: str, target_ids: set, model, k: int = SEARCH_K):
    """Return the best (lowest) rank at which any target chunk id appears, or None."""
    qvec = model.encode([query], normalize_embeddings=True)
    qvec = np.array(qvec, dtype="float32")
    scores, indices = index.search(qvec, k)

    for rank, idx in enumerate(indices[0], 1):
        chunk = metadata[idx]
        if chunk["id"] in target_ids:
            return rank, float(scores[0][rank - 1])
    return None, None


def main():
    print("Loading embedding model and index...")
    model = SentenceTransformer("BAAI/bge-m3")
    index = faiss.read_index(str(CHUNKS_DIR / "faiss.index"))
    metadata = json.loads((CHUNKS_DIR / "chunk_metadata.json").read_text())

    entries = [json.loads(l) for l in EVAL_SET_FILE.read_text(encoding="utf-8").splitlines()]
    answerable = [e for e in entries if e["answerable"] and e.get("source_chunk_ids")]

    print(f"{len(answerable)} answerable queries with known source chunks.\n")

    results = []
    for entry in answerable:
        query = entry["query"]
        target_ids = set(entry["source_chunk_ids"])
        normalized = normalize_roman_urdu(query)

        raw_rank, raw_score = find_rank(index, metadata, query, target_ids, model)
        norm_rank, norm_score = find_rank(index, metadata, normalized, target_ids, model)

        changed = normalized != query
        improved = (raw_rank is None and norm_rank is not None) or (
            raw_rank is not None and norm_rank is not None and norm_rank < raw_rank
        )

        result = {
            "query": query,
            "normalized": normalized,
            "text_changed": changed,
            "raw_rank": raw_rank,
            "raw_score": round(raw_score, 3) if raw_score else None,
            "normalized_rank": norm_rank,
            "normalized_score": round(norm_score, 3) if norm_score else None,
            "improved": improved,
        }
        results.append(result)

        print(f"{query[:50]}")
        print(f"  raw rank: {raw_rank}  |  normalized rank: {norm_rank}  |  changed: {changed}  |  improved: {improved}")
        print()

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    n_changed = sum(1 for r in results if r["text_changed"])
    n_improved = sum(1 for r in results if r["improved"])
    n_found_raw = sum(1 for r in results if r["raw_rank"] is not None)
    n_found_norm = sum(1 for r in results if r["normalized_rank"] is not None)

    print("=" * 60)
    print(f"Queries where normalization changed the text: {n_changed}/{len(results)}")
    print(f"Queries where normalization improved rank: {n_improved}/{len(results)}")
    print(f"Found in top-{SEARCH_K} (raw): {n_found_raw}/{len(results)}")
    print(f"Found in top-{SEARCH_K} (normalized): {n_found_norm}/{len(results)}")
    print(f"\nFull results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
