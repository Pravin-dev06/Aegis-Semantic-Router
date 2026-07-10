# Aegis Semantic Router

> **A zero-token local-first hybrid AI agent gateway for the AMD Developer Hackathon ACT II — Track 1: General-Purpose AI Agent.**

Aegis intelligently routes natural language queries between a fully **in-process local Gemma-2-2B-Instruct model** (zero API cost) and a premium **Fireworks AI** cloud model. The routing decision is made by a CPU-only semantic kNN classifier that compares incoming prompts against a curated reference corpus covering all 8 hackathon evaluation categories.

The result: **maximum accuracy at minimum Fireworks token spend**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Routing Categories](#routing-categories)
- [How The Semantic Router Works](#how-the-semantic-router-works)
- [Project Structure](#project-structure)
- [Quick Start (Local Development)](#quick-start-local-development)
- [Docker Deployment](#docker-deployment)
- [Simulating the Judge's Environment](#simulating-the-judges-environment)
- [Harness Mode (How Judges Run It)](#harness-mode-how-judges-run-it)
- [Web Dashboard](#web-dashboard)
- [Configuration Reference](#configuration-reference)
- [API Endpoints](#api-endpoints)
- [Running Tests](#running-tests)
- [License](#license)

---

## Overview

| Property | Value |
|---|---|
| **Local Model** | `gemma-2-2b-it-Q4_K_M.gguf` (in-process via `llama-cpp-python`) |
| **Router Model** | `all-MiniLM-L6-v2` (Hugging Face, CPU-only, ~100 MB) |
| **Remote Provider** | Fireworks AI (env-configurable via `ALLOWED_MODELS`) |
| **Routing Strategy** | Semantic kNN (k=7) cosine similarity |
| **Reference Corpus** | 40 prompts — 5 local + 5 remote × 8 hackathon categories |
| **RAM Usage** | ~2.5 GB peak (fits within judging VM's 4 GB limit) |
| **Container Startup** | < 15 seconds (all model weights pre-baked at build time) |
| **Offline Runtime** | ✅ Yes — no internet access needed at runtime |

---

## Architecture

```
                        ┌─────────────────────────────────┐
                        │        User / Judging Harness   │
                        └──────────────┬──────────────────┘
                                       │ prompt
                                       ▼
                        ┌─────────────────────────────────┐
                        │    Aegis Semantic Router         │
                        │                                  │
                        │  1. Embed prompt (CPU, ~10ms)   │
                        │     all-MiniLM-L6-v2            │
                        │                                  │
                        │  2. kNN cosine similarity        │
                        │     against 40 reference prompts │
                        │                                  │
                        │  3. Compute remote_vote_fraction │
                        │     score = votes / k  (0.0–1.0) │
                        │                                  │
                        │  4. Compare score to threshold   │
                        │     (per-category overrides)     │
                        └─────────────┬───────────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               │ score ≤ threshold                            │ score > threshold
               ▼                                             ▼
  ┌────────────────────────┐                   ┌────────────────────────┐
  │    Local Provider       │                   │    Remote Provider      │
  │  gemma-2-2b-it Q4_K_M  │                   │    Fireworks AI API     │
  │  llama-cpp-python       │                   │  (from ALLOWED_MODELS)  │
  │  Cost: 0 tokens ✅      │                   │  Cost: counted tokens   │
  └────────────┬───────────┘                   └────────────────────────┘
               │ on exception
               ▼
  ┌────────────────────────┐
  │  Auto-Fallback →        │
  │  Remote Provider        │
  └────────────────────────┘
```

---

## Routing Categories

The semantic router is tuned to handle all 8 official hackathon evaluation categories:

| # | Category | Default Route | Threshold | Reason |
|---|---|---|---|---|
| 1 | Factual Knowledge | **Local** | 0.43 | Simple recall — Gemma 2B handles well |
| 2 | Mathematical Reasoning | **Remote** | 0.35 | Multi-step arithmetic needs precision |
| 3 | Sentiment Classification | **Local** | 0.55 | Gemma 2B excels at basic classification |
| 4 | Text Summarisation | **Local** | 0.55 | Condensing is well within 2B capability |
| 5 | Named Entity Recognition | **Local** | 0.55 | Structured extraction stays local |
| 6 | Code Debugging | **Remote** | 0.35 | Subtle bugs need deep reasoning |
| 7 | Logical / Deductive Reasoning | **Remote** | 0.35 | Constraint puzzles need stronger models |
| 8 | Code Generation | **Remote** | 0.35 | Correctness matters — remote wins |

---

## How The Semantic Router Works

The router uses a **cosine similarity kNN classifier** with k=7 to decide whether a prompt should go to the local or remote model.

### Reference Corpus
A balanced corpus of **40 reference prompts** is hand-curated:
- **5 "local" examples** per category: simple, single-step queries well within the capability of Gemma-2-2B.
- **5 "remote" examples** per category: complex, multi-step queries that require a stronger model.

### Scoring
1. The incoming prompt is embedded into a 384-dimensional vector using `all-MiniLM-L6-v2` (runs in-process on CPU).
2. Cosine similarity is computed against all 40 reference embeddings (pre-computed at startup).
3. The top-7 nearest neighbours are selected.
4. The **remote vote fraction** = number of "remote" neighbours / 7.
5. If the fraction exceeds the category threshold → route **remote**. Otherwise → route **local**.

### Graceful Fallback
If `sentence-transformers` is unavailable (e.g., during unit tests without model weights), the router automatically falls back to a keyword + length **heuristic scorer** to guarantee routing decisions are always made.

---

## Project Structure

```
aegis-semantic-router/
│
├── app.py                  # FastAPI web server + CLI entry point + harness loop
├── router.py               # Semantic kNN router, heuristic fallback, routing result
├── providers.py            # LocalProvider (llama-cpp-python) + RemoteProvider (Fireworks)
├── config.py               # Pydantic config models + environment variable loading
├── config.yaml             # All routing thresholds + model paths (no secrets)
├── benchmark.py            # Benchmark runner and accuracy evaluator
├── verify_local.py         # One-command local validation pipeline (deps + tests + harness)
│
├── requirements.txt        # Pinned CPU-only PyTorch + sentence-transformers
├── Dockerfile              # Bakes both model weights into image at build time
├── .dockerignore           # Excludes venv/, models/, input/, output/, spec/
├── .gitignore              # Excludes secrets, model weights, runtime I/O
│
├── static/                 # Web dashboard HTML, CSS, and JavaScript
├── datasets/               # Benchmark evaluation datasets (sample.json)
├── tests/                  # 42 unit tests (routing, config, providers, benchmark)
└── models/                 # Local GGUF weights staging (git-ignored, docker-ignored)
```

---

## Quick Start (Local Development)

### 1. Clone and set up the virtual environment

```bash
git clone https://github.com/your_username/aegis-semantic-router.git
cd aegis-semantic-router

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
# Open .env and set:
# FIREWORKS_API_KEY=your_fireworks_api_key_here
```

### 3. Run the automated verifier

This script checks your environment, downloads the local model if missing, runs all unit tests, and simulates the judging harness end-to-end:

```bash
python verify_local.py
```

Expected final output:
```
[SUCCESS] All unit tests passed!
[SUCCESS] results.json matches the REQUIRED evaluation schema exactly!
CONGRATULATIONS! Your local router setup is fully verified and clean.
```

---

## Docker Deployment

### Build the container
The Docker build automatically:
1. Installs all CPU-optimised Python dependencies (no CUDA).
2. Downloads and bakes `all-MiniLM-L6-v2` (router model, ~100 MB) into the image.
3. Downloads and bakes `gemma-2-2b-it-Q4_K_M.gguf` (local inference model, ~1.6 GB) into the image.
4. Sets `TRANSFORMERS_OFFLINE=1` so the container runs with **zero runtime internet access**.

```bash
# Build for the judging VM architecture (linux/amd64)
docker build --platform linux/amd64 -t your_dockerhub_username/amd-ai-router:latest .
```

### Push to Docker Hub

```bash
docker login
docker push your_dockerhub_username/amd-ai-router:latest
```

---

## Simulating the Judge's Environment

Run the published image locally with **identical hardware constraints** to the judging VM (4 GB RAM, 2 vCPUs):

```powershell
docker run --rm `
  --memory="4g" `
  --cpus="2" `
  -v "${pwd}/input:/input" `
  -v "${pwd}/output:/output" `
  -e FIREWORKS_API_KEY="your_fireworks_api_key" `
  -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" `
  -e ALLOWED_MODELS="accounts/fireworks/models/gemma2-2b-it,accounts/fireworks/models/gemma2-27b-it" `
  your_dockerhub_username/amd-ai-router:latest
```

After the run, inspect `output/results.json` on your host machine. It should contain one entry per task with `task_id` and `answer` fields.

---

## Harness Mode (How Judges Run It)

When the container starts, `app.py` checks for the environment variable `TASKS_JSON_PATH` (or the default path `/input/tasks.json`). If the file is found, the app enters **Harness Mode** automatically:

### Input file: `/input/tasks.json`

```json
[
  { "task_id": "task-001", "prompt": "What is the boiling point of water?" },
  { "task_id": "task-002", "prompt": "Solve: 2x + 3y = 12 and x - y = 1." },
  { "task_id": "task-003", "prompt": "Write a Python function to reverse a string." }
]
```

### Output file: `/output/results.json`

```json
[
  { "task_id": "task-001", "answer": "Water boils at 100°C (212°F) at sea level." },
  { "task_id": "task-002", "answer": "x = 3, y = 2." },
  { "task_id": "task-003", "answer": "def reverse_string(s): return s[::-1]" }
]
```

### Harness Runtime Behaviour
1. Loads config from `config.yaml` + environment variable overrides.
2. Calls `preload_semantic_model()` to warm up the router before processing begins.
3. Processes all tasks **concurrently** using `asyncio.gather`.
4. On any per-task exception, writes `""` as the answer and logs the error (prevents a single task failure from crashing the entire run).
5. Writes `results.json` and exits cleanly.

---

## Web Dashboard

If the container starts **without** `/input/tasks.json`, it boots a FastAPI web server on port `8000`:

```bash
python app.py
# Open http://localhost:8000
```

### Dashboard Features
- **Live Prompt Router**: Submit prompts and instantly view the routing decision, complexity score, category detected, provider used, model used, and latency.
- **Configuration Inspector**: View all active threshold and model configuration parameters.
- **Routing Logs**: Real-time view of how each prompt was classified and why.

---

## Configuration Reference

All non-secret settings live in `config.yaml`. Secrets (API keys) are passed via environment variables.

### `config.yaml`

```yaml
local_provider:
  base_url: "http://localhost:1234/v1"   # Only used in local dev without llama-cpp
  model: "models/gemma-2-2b-it-Q4_K_M.gguf"
  timeout: 30

remote_provider:
  base_url: "https://api.fireworks.ai/inference/v1"
  model: "gemma-4-31b"
  timeout: 60

routing:
  threshold: 0.5             # Global fallback threshold
  strategy: semantic         # Options: semantic | heuristic | gatekeeper
  fallback_enabled: true

  category_thresholds:
    mathematics: 0.35        # needs ≥4/7 remote votes → escalate on complex maths
    programming: 0.35        # needs ≥4/7 remote votes → escalate on real code tasks
    logical_reasoning: 0.35  # needs ≥4/7 remote votes → escalate on constraint puzzles
    summarization: 0.55      # needs ≥5/7 remote votes → trust local for most summaries
    classification: 0.55     # needs ≥5/7 remote votes → trust local for sentiment
    ner: 0.55                # needs ≥5/7 remote votes → trust local for simple NER
    general: 0.43            # global midpoint — balanced default
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `FIREWORKS_API_KEY` | ✅ Yes | Your Fireworks AI API key |
| `FIREWORKS_BASE_URL` | Optional | Override the Fireworks API endpoint |
| `ALLOWED_MODELS` | Optional | Comma-separated list of models (first=local, second=remote) |
| `TASKS_JSON_PATH` | Optional | Override the default `/input/tasks.json` path |
| `RESULTS_JSON_PATH` | Optional | Override the default `/output/results.json` path |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web dashboard UI |
| `GET` | `/api/config` | Returns current active configuration |
| `POST` | `/api/route` | Routes a prompt and returns the full routing decision |

### `POST /api/route`

**Request:**
```json
{
  "prompt": "Write a Python function to implement binary search.",
  "threshold": 0.5
}
```

**Response:**
```json
{
  "provider_used": "remote",
  "model_used": "accounts/fireworks/models/gemma2-27b-it",
  "complexity_score": 0.71,
  "routing_reason": "semantic kNN k=7 votes=5/7 category=remote near='Write a Python class implementing a binary search tree...'",
  "category": "programming",
  "threshold_used": 0.35,
  "fallback_used": false,
  "latency_sec": 0.012,
  "response_content": "def binary_search(arr, target): ..."
}
```

---

## Running Tests

```bash
# Run the full test suite (42 tests)
pytest tests/ -v

# Run a specific test file
pytest tests/test_router.py -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=term-missing
```

---

## License

MIT License

Copyright (c) 2025 Pravin 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
