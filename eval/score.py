"""
score.py — Run the eval set through the full RAG pipeline and score results.

Uses BGE-M3 embedding similarity (not literal word overlap) to check whether
each gold fact's meaning is present in the generated answer — necessary since
answers are often in Roman Urdu / mixed language while gold facts are recorded
in English. Saves incrementally after every query (crash-safe, resumable).

Usage:
    python eval/score.py --model-tag q4_k_m
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from rag import RAGPipeline  # noqa: E402

BASE = Path.home() / "Downloads/Projects/manar-prep"
EVAL_SET_FILE = BASE / "eval/eval_set.jsonl"
RESULTS_DIR = BASE / "eval/results"

# Similarity threshold above which a gold fact counts as "present" in the answer.
# Calibrate by eyeballing a few known-correct vs known-wrong pairs.
FACT_MATCH_THRESHOLD = 0.55

REFUSAL_MARKERS = [
    # English
    "don't have that information",
    "not mentioned",
    "no information",
    "cannot find",
    "not covered",
    "not provided in the",
    "does not mention",
    "context does not",
    "not specified",
    # Roman Urdu
    "koi information nahi",
    "koi jawab nahi",
    "nahi hai context",
    "context mein nahi",
    "maloomat nahi",
    "koi maloomat nahi",
]


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def is_refusal(answer: str) -> bool:
    answer_norm = normalize(answer)
    return any(marker in answer_norm for marker in REFUSAL_MARKERS)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def score_entry(entry: dict, answer: str, embed_model) -> dict:
    if not entry["answerable"]:
        correct = is_refusal(answer)
        return {
            "query": entry["query"],
            "type": "refusal_check",
            "correct_refusal": correct,
            "score": 1.0 if correct else 0.0,
            "answer": answer,
        }
    else:
        facts = entry["gold_facts"]
        if not facts:
            return {
                "query": entry["query"], "type": "fact_match",
                "gold_facts": [], "facts_matched": [], "fact_similarities": [],
                "score": 0.0, "answer": answer,
            }

        # Embed the answer once, and each gold fact, then compare
        answer_vec = embed_model.encode([answer], normalize_embeddings=True)[0]
        fact_vecs = embed_model.encode(facts, normalize_embeddings=True)

        similarities = [cosine_sim(answer_vec, fv) for fv in fact_vecs]
        matched = [sim >= FACT_MATCH_THRESHOLD for sim in similarities]
        score = sum(matched) / len(facts)

        return {
            "query": entry["query"],
            "type": "fact_match",
            "gold_facts": facts,
            "facts_matched": matched,
            "fact_similarities": [round(s, 3) for s in similarities],
            "score": score,
            "answer": answer,
        }


def load_partial_results(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Don't treat errored/timed-out queries as "done" -- retry them
        return {
            r["query"]: r for r in data.get("results", [])
            if "[ERROR" not in r.get("answer", "")
        }
    except (json.JSONDecodeError, KeyError):
        return {}


def write_summary(model_tag: str, results: list[dict], out_path: Path):
    fact_scores = [r["score"] for r in results if r["type"] == "fact_match"]
    refusal_scores = [r["score"] for r in results if r["type"] == "refusal_check"]
    summary = {
        "model_tag": model_tag,
        "n_total": len(results),
        "n_answerable": len(fact_scores),
        "n_not_answerable": len(refusal_scores),
        "mean_fact_match_score": (sum(fact_scores) / len(fact_scores)) if fact_scores else None,
        "correct_refusal_rate": (sum(refusal_scores) / len(refusal_scores)) if refusal_scores else None,
        "results": results,
    }
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--rescore-only", action="store_true",
                         help="Re-score existing saved answers without re-querying the model")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"results_{args.model_tag}.json"

    entries = [json.loads(l) for l in EVAL_SET_FILE.read_text(encoding="utf-8").splitlines()]
    print(f"Loaded {len(entries)} eval entries.")

    print("Loading embedding model for scoring...")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("BAAI/bge-m3")

    if args.rescore_only:
        # Re-score previously saved answers without hitting llama-server again
        prior = load_partial_results(out_path)
        if not prior:
            print("No prior results found to rescore.")
            return
        results = []
        for entry in entries:
            prev = prior.get(entry["query"])
            if not prev or prev.get("answer", "").startswith("[ERROR"):
                continue
            result = score_entry(entry, prev["answer"], embed_model)
            result["elapsed_sec"] = prev.get("elapsed_sec", 0)
            results.append(result)
            print(f"  {entry['query'][:50]}  score={result['score']:.2f}")
        write_summary(args.model_tag, results, out_path)
        summary = json.loads(out_path.read_text())
        print("\n" + "=" * 60)
        print(f"Rescored. Mean fact-match: {summary['mean_fact_match_score']:.2f}"
              if summary['mean_fact_match_score'] is not None else "N/A")
        return

    done = load_partial_results(out_path)
    if done:
        print(f"Found {len(done)} previously-scored queries — resuming.")

    print("Initializing RAG pipeline...")
    rag = RAGPipeline()

    results = list(done.values())

    for i, entry in enumerate(entries, 1):
        if entry["query"] in done:
            print(f"[{i}/{len(entries)}] SKIP (already scored): {entry['query'][:50]}")
            continue

        print(f"\n[{i}/{len(entries)}] {entry['query'][:60]}")
        t0 = time.time()
        try:
            resp = rag.answer(entry["query"], verbose=False)
            answer = resp["answer"]
        except Exception as e:
            print(f"  ERROR: {e}")
            answer = f"[ERROR: {e}]"
        elapsed = time.time() - t0

        result = score_entry(entry, answer, embed_model)
        result["elapsed_sec"] = round(elapsed, 1)
        results.append(result)

        print(f"  score={result['score']:.2f}  ({elapsed:.1f}s)")
        write_summary(args.model_tag, results, out_path)

    summary = write_summary(args.model_tag, results, out_path)

    print("\n" + "=" * 60)
    print(f"Model: {args.model_tag}")
    if summary["mean_fact_match_score"] is not None:
        print(f"Mean fact-match score (answerable): {summary['mean_fact_match_score']:.2f}")
    if summary["correct_refusal_rate"] is not None:
        print(f"Correct refusal rate (not-answerable): {summary['correct_refusal_rate']:.2f}")
    print(f"Full results: {out_path}")


if __name__ == "__main__":
    main()
