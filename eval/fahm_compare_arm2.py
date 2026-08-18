"""
fahm_compare_arm2.py — Compare retrieval rank: raw Roman Urdu vs. translated-to-English.

Adapted FAHM Arm 2 (translation, not transliteration -- see fahm_translate.py
for reasoning). Requires llama-server running for translation calls.
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from fahm_translate import translate_to_english  # noqa: E402

BASE = Path.home() / "Downloads/Projects/manar-prep"
CHUNKS_DIR = BASE / "corpus/chunks"
EVAL_SET_FILE = BASE / "eval/eval_set.jsonl"
RESULTS_FILE = BASE / "eval/results/fahm_arm2_comparison.json"

SEARCH_K = 20


def find_rank(index, metadata, query: str, target_ids: set, model, k: int = SEARCH_K):
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

        print(f"Translating: {query[:50]}...")
        try:
            translated = translate_to_english(query)
        except Exception as e:
            print(f"  Translation error: {e}")
            translated = None

        raw_rank, raw_score = find_rank(index, metadata, query, target_ids, model)
        trans_rank, trans_score = (None, None)
        if translated:
            trans_rank, trans_score = find_rank(index, metadata, translated, target_ids, model)

        improved = (raw_rank is None and trans_rank is not None) or (
            raw_rank is not None and trans_rank is not None and trans_rank < raw_rank
        )

        result = {
            "query": query,
            "translated": translated,
            "raw_rank": raw_rank,
            "raw_score": round(raw_score, 3) if raw_score else None,
            "translated_rank": trans_rank,
            "translated_score": round(trans_score, 3) if trans_score else None,
            "improved": improved,
        }
        results.append(result)

        print(f"  EN: {translated}")
        print(f"  raw rank: {raw_rank}  |  translated rank: {trans_rank}  |  improved: {improved}\n")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    n_improved = sum(1 for r in results if r["improved"])
    n_found_raw = sum(1 for r in results if r["raw_rank"] is not None)
    n_found_trans = sum(1 for r in results if r["translated_rank"] is not None)

    print("=" * 60)
    print(f"Queries where translation improved rank: {n_improved}/{len(results)}")
    print(f"Found in top-{SEARCH_K} (raw): {n_found_raw}/{len(results)}")
    print(f"Found in top-{SEARCH_K} (translated): {n_found_trans}/{len(results)}")
    print(f"\nFull results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
