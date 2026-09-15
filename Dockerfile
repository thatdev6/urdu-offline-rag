# --- Stage 1: build llama.cpp's server binary ---
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
WORKDIR /build/llama.cpp
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
    && cmake --build build --config Release -j$(nproc) --target llama-server

# --- Stage 2: the actual runtime image ---
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

ARG HF_MODEL_REPO=alchemistVI/urdu-offline-rag-qwen3-4b-gguf
ENV HF_MODEL_REPO=${HF_MODEL_REPO}

RUN mkdir -p /app/models/gguf && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='${HF_MODEL_REPO}', filename='qwen3-4b-q4_k_m.gguf', local_dir='/app/models/gguf')"

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY pipeline/ ./pipeline/
COPY corpus/chunks/ ./corpus/chunks/
COPY app.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV PORT=8080
ENV LLAMA_SERVER_URL=http://localhost:8081/v1/chat/completions

EXPOSE 8080

ENTRYPOINT ["./entrypoint.sh"]
