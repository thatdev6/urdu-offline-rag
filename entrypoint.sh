#!/bin/bash
set -e

MODEL_PATH="/app/models/gguf/qwen3-4b-q4_k_m.gguf"

echo "Starting llama-server..."
llama-server -m "$MODEL_PATH" -c 3072 --parallel 1 --port 8081 &

echo "Starting app..."
exec python app.py
