"""Application entry point.

Handles both:
1. Command Line Interface (CLI): routing prompts and running evaluation benchmarks.
2. Web API Dashboard: serving FastAPI server for visual interactive prompt routing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from benchmark import run_benchmark
from config import load_config
from providers import LocalProvider, RemoteProvider
from router import route, preload_semantic_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-14s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")


# ---------------------------------------------------------------------------
# FastAPI Server Application Setup (Web Mode)
# ---------------------------------------------------------------------------

class RouteRequest(BaseModel):
    """Payload format for the route POST API endpoint."""
    prompt: str
    threshold: Optional[float] = None


class RouteResponse(BaseModel):
    """Response structure returned to the frontend dashboard."""
    provider_used: str
    model_used: str
    complexity_score: float
    routing_reason: str
    fallback_used: bool
    latency_sec: float
    response_content: str


# Global instances initialized during FastAPI lifespan
local_provider_instance: Optional[LocalProvider] = None
remote_provider_instance: Optional[RemoteProvider] = None
global_config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize system configuration and providers on FastAPI startup."""
    global local_provider_instance, remote_provider_instance, global_config
    logger.info("Initializing AMD AI Router Web Services...")
    try:
        global_config = load_config()
        local_provider_instance = LocalProvider(global_config.local_provider)
        remote_provider_instance = RemoteProvider(global_config.remote_provider)
        # Pre-load the semantic router model so the first request is not slow
        preload_semantic_model()
        logger.info("Web dashboard services fully initialized.")
    except Exception as e:
        logger.critical("Startup failed. Config or Provider setup error: %s", e)
        raise e
    yield


app = FastAPI(
    title="AMD Adaptive AI Router Dashboard",
    description="Track 1 submission demo interface.",
    lifespan=lifespan
)


# Serve static assets (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_dashboard():
    """Serve the main HTML dashboard interface."""
    return FileResponse("static/index.html")


@app.get("/api/config")
async def get_config():
    """Expose current model names and threshold to the frontend."""
    if not global_config:
        raise HTTPException(status_code=500, detail="Config not loaded")
    return {
        "local_model": global_config.local_provider.model,
        "remote_model": global_config.remote_provider.model,
        "threshold": global_config.routing.threshold,
    }


@app.post("/api/route", response_model=RouteResponse)
async def api_route_prompt(payload: RouteRequest):
    """Route prompt asynchronously and return results to the UI dashboard."""
    if not local_provider_instance or not remote_provider_instance or not global_config:
        raise HTTPException(status_code=500, detail="Providers not initialized")
    
    threshold = payload.threshold if payload.threshold is not None else global_config.routing.threshold
    
    start_time = time.perf_counter()
    try:
        result = await route(
            prompt=payload.prompt,
            local_provider=local_provider_instance,
            remote_provider=remote_provider_instance,
            threshold=threshold,
            fallback_enabled=global_config.routing.fallback_enabled,
            category_thresholds=global_config.routing.category_thresholds,
            strategy=global_config.routing.strategy,
        )
        latency = time.perf_counter() - start_time
        
        return RouteResponse(
            provider_used=result.provider_used,
            model_used=result.model_used,
            complexity_score=result.complexity_score,
            routing_reason=result.routing_reason,
            fallback_used=result.fallback_used,
            latency_sec=latency,
            response_content=result.response.content,
        )
    except Exception as e:
        logger.error("API prompt routing failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))



# ---------------------------------------------------------------------------
# Submission Harness Task Processor (Evaluation Mode)
# ---------------------------------------------------------------------------

async def run_harness_mode(tasks_path: str) -> None:
    """Read tasks.json → route each prompt → write results.json.

    This mode is triggered automatically when the evaluation harness mounts
    a tasks file at /input/tasks.json (or TASKS_JSON_PATH env var).

    Input format:  [{"task_id": "t1", "prompt": "..."}, ...]
    Output format: [{"task_id": "t1", "answer": "..."}, ...]
    """
    results_path = os.getenv("RESULTS_JSON_PATH", "/output/results.json")
    logger.info("=== HARNESS MODE: reading tasks from %s ===", tasks_path)

    # Pre-load the semantic router model before any tasks are processed
    preload_semantic_model()

    try:
        config = load_config()
    except Exception as e:
        logger.critical("Config load failed in harness mode: %s", e)
        sys.exit(1)

    local = LocalProvider(config.local_provider)
    remote = RemoteProvider(config.remote_provider)

    try:
        with open(tasks_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as e:
        logger.critical("Failed to read tasks file %s: %s", tasks_path, e)
        sys.exit(1)

    if not isinstance(tasks, list):
        logger.critical("tasks.json must be a JSON list of task objects")
        sys.exit(1)

    logger.info("Loaded %d tasks", len(tasks))
    results: list[dict] = []
    semaphore = asyncio.Semaphore(5)   # Max 5 concurrent provider calls

    async def _process(task: dict) -> None:
        task_id = task.get("task_id") or task.get("id")
        prompt  = task.get("prompt", "")
        if not task_id:
            logger.warning("Skipping task without task_id: %s", task)
            return
        async with semaphore:
            try:
                r = await route(
                    prompt=prompt,
                    local_provider=local,
                    remote_provider=remote,
                    threshold=config.routing.threshold,
                    fallback_enabled=config.routing.fallback_enabled,
                    category_thresholds=config.routing.category_thresholds,
                    strategy=config.routing.strategy,
                )
                results.append({"task_id": task_id, "answer": r.response.content})
                logger.info("task %-10s → %-6s (%s)", task_id, r.provider_used, r.routing_reason)
            except Exception as ex:
                logger.error("task %s failed: %s", task_id, ex)
                results.append({"task_id": task_id, "answer": ""})

    await asyncio.gather(*[_process(t) for t in tasks])

    # Write output — ensure parent directory exists
    try:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Wrote %d results to %s", len(results), results_path)
    except Exception as e:
        logger.critical("Failed to write results file: %s", e)
        sys.exit(1)

    logger.info("=== HARNESS MODE COMPLETE ===")


# ---------------------------------------------------------------------------
# Command Line Interface Setup (CLI Mode)
# ---------------------------------------------------------------------------

async def run_cli() -> None:
    """CLI logic for running prompts or benchmarks directly from terminal."""
    parser = argparse.ArgumentParser(
        description="AMD AI Router CLI tool. Route prompts or run evaluation benchmarks."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="The input prompt to route to the appropriate model.",
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        action="store_true",
        help="Run the evaluation benchmark on the default sample dataset.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default="datasets/sample.json",
        help="Path to a custom JSON dataset for running the benchmark (default: datasets/sample.json).",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        help="Override the complexity threshold (0.0 to 1.0) configured in config.yaml.",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config()
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load configuration: %s", e)
        sys.exit(1)

    threshold = args.threshold if args.threshold is not None else config.routing.threshold

    # Initialize providers
    local = LocalProvider(config.local_provider)
    remote = RemoteProvider(config.remote_provider)

    # 1. Benchmark execution path
    if args.benchmark:
        logger.info("Executing benchmark mode...")
        try:
            await run_benchmark(
                dataset_path=args.dataset,
                local_provider=local,
                remote_provider=remote,
                threshold=threshold,
                fallback_enabled=config.routing.fallback_enabled,
                category_thresholds=config.routing.category_thresholds,
                strategy=config.routing.strategy,
            )
        except Exception as e:
            logger.error("Benchmark execution failed: %s", e)
            sys.exit(1)
        return

    # 2. Single prompt routing path
    if args.prompt:
        logger.info("Routing prompt: '%s'", args.prompt)
        try:
            start_time = time.perf_counter()
            result = await route(
                prompt=args.prompt,
                local_provider=local,
                remote_provider=remote,
                threshold=threshold,
                fallback_enabled=config.routing.fallback_enabled,
                category_thresholds=config.routing.category_thresholds,
                strategy=config.routing.strategy,
            )
            latency = time.perf_counter() - start_time
            logger.info("Routing completed successfully!")
            print("\n" + "=" * 50)
            print("                 ROUTING RESULTS")
            print("=" * 50)
            print(f"Provider Used:    {result.provider_used.upper()}")
            print(f"Model Used:       {result.model_used}")
            print(f"Complexity Score: {result.complexity_score:.2f} (threshold: {threshold:.2f})")
            print(f"Decision Reason:  {result.routing_reason}")
            print(f"Fallback Used:    {result.fallback_used}")
            print(f"Latency:          {latency:.3f}s")
            print(f"Response Content:\n{result.response.content}")
            print("=" * 50 + "\n")
        except Exception as e:
            logger.error("Failed to route prompt: %s", e)
            sys.exit(1)
        return

    # 3. Default fallback if no action specified
    parser.print_help()


# ---------------------------------------------------------------------------
# Main Router Execution Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Priority 1: Evaluation harness mode — /input/tasks.json auto-detected
    tasks_path = os.getenv("TASKS_JSON_PATH", "/input/tasks.json")
    if os.path.exists(tasks_path):
        asyncio.run(run_harness_mode(tasks_path))
        sys.exit(0)

    # Priority 2: CLI mode — user passed arguments
    if len(sys.argv) > 1:
        asyncio.run(run_cli())

    # Priority 3: Web dashboard mode — no args, launch FastAPI server
    else:
        import uvicorn
        logger.info("Starting uvicorn server in web dashboard mode...")
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
