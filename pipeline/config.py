"""
config.py -- central configuration for the pipeline and eval scripts.

Paths default to locations relative to this file, meaning relative to the
repo root, so the repo works immediately for anyone who clones it. Any
value here can be overridden by setting it in a .env file at the repo root
(see .env.example) or as a real environment variable, without needing to
edit any script.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv():
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _path(env_key: str, default: Path) -> Path:
    return Path(os.environ.get(env_key, str(default)))


# corpus
CORPUS_DIR = _path("CORPUS_DIR", REPO_ROOT / "corpus")
CORPUS_RAW = CORPUS_DIR / "raw"
CORPUS_CLEAN = CORPUS_DIR / "clean"
CORPUS_CHUNKS = CORPUS_DIR / "chunks"
CHUNKS_FILE = CORPUS_CHUNKS / "chunks.jsonl"
INDEX_FILE = CORPUS_CHUNKS / "faiss.index"
METADATA_FILE = CORPUS_CHUNKS / "chunk_metadata.json"

# eval
EVAL_DIR = _path("EVAL_DIR", REPO_ROOT / "eval")
EVAL_SET_FILE = EVAL_DIR / "eval_set.jsonl"
QUERIES_RAW_FILE = EVAL_DIR / "queries_raw.txt"
RESULTS_DIR = EVAL_DIR / "results"

# models -- large, gitignored, often lives outside the repo entirely.
# override MODELS_DIR in .env if yours isn't inside the repo folder.
MODELS_DIR = _path("MODELS_DIR", REPO_ROOT / "models")

# runtime settings
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080/v1/chat/completions")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "BAAI/bge-m3")
TOP_K = int(os.environ.get("TOP_K", 12))
FACT_MATCH_THRESHOLD = float(os.environ.get("FACT_MATCH_THRESHOLD", 0.55))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", 900))
