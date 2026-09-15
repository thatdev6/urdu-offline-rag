#!/bin/bash
set -e

HF_MODEL_REPO="${HF_MODEL_REPO:-alchemistVI/urdu-offline-rag-qwen3-4b-gguf}"
MODEL_PATH="/app/models/gguf/qwen3-4b-q4_k_m.gguf"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Downloading model from ${HF_MODEL_REPO}..."
  mkdir -p /app/models/gguf
  python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='${HF_MODEL_REPO}',
    filename='qwen3-4b-q4_k_m.gguf',
    local_dir='/app/models/gguf'
)
"
fi

echo "Starting llama-server..."
llama-server -m "$MODEL_PATH" -c 3072 --parallel 1 --port 8081 &

until curl -s http://localhost:8081/health > /dev/null 2>&1; do
  echo "Waiting for llama-server..."
  sleep 2
done

echo "Starting app..."
exec python app.py
