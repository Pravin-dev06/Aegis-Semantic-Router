FROM python:3.11-slim

WORKDIR /app

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: System dependencies
# wget is used to download the GGUF model and then removed to save space.
# ─────────────────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Bake the semantic ROUTER model into the image (~100 MB).
#
# all-MiniLM-L6-v2 is the sentence-transformers embedding model used by the
# kNN classifier to route prompts to local vs. remote providers.
# Downloading at build time means the container starts fully offline —
# no runtime HuggingFace requests that could hang under the 60-second limit.
# ─────────────────────────────────────────────────────────────────────────────
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
print('Downloading router model: all-MiniLM-L6-v2 ...'); \
m = SentenceTransformer('all-MiniLM-L6-v2'); \
print(f'Router model ready. Embedding dim: {m.get_sentence_embedding_dimension()}') \
"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Bake the LOCAL INFERENCE model into the image (~2.0 GB).
#
# qwen2.5-3b-instruct-q4_k_m.gguf runs via llama-cpp-python entirely in-process.
# Local answers count fully toward accuracy and cost 0 Fireworks tokens.
# Fits comfortably within the 4 GB RAM / 2 vCPU grading environment.
# ─────────────────────────────────────────────────────────────────────────────
RUN mkdir -p /app/models \
    && wget -q --show-progress \
       -O /app/models/qwen2.5-3b-instruct-q4_k_m.gguf \
       https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf

# Remove wget now that all downloads are complete
RUN apt-get purge -y wget && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Force offline mode at runtime.
#
# Both models are baked in above. Setting these variables ensures the
# container never tries to reach HuggingFace Hub at runtime, which would
# silently hang and trigger a TIMEOUT failure from the judging harness.
# ─────────────────────────────────────────────────────────────────────────────
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: Copy application source code
# ─────────────────────────────────────────────────────────────────────────────
COPY . .

# Run the application
# On startup app.py checks for /input/tasks.json (harness mode) first,
# then falls back to CLI or web dashboard mode.
CMD ["python", "app.py"]
