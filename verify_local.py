"""Local Verification and Validation Script for AMD AI Router.

This script checks dependencies, model weights, runs tests, and simulates
the evaluation harness to ensure your submission container is ready.
"""

import sys
import os
import json
import subprocess
from pathlib import Path
import urllib.request

def print_status(message, status="INFO"):
    colors = {
        "INFO": "\033[94m[INFO]\033[0m",
        "SUCCESS": "\033[92m[SUCCESS]\033[0m",
        "WARNING": "\033[93m[WARNING]\033[0m",
        "ERROR": "\033[91m[ERROR]\033[0m"
    }
    print(f"{colors.get(status, '[INFO]')} {message}")

def check_venv():
    # Simple check for virtual environment
    is_venv = (
        hasattr(sys, "real_prefix") or
        (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    )
    if is_venv:
        print_status("Running inside a virtual environment.", "SUCCESS")
    else:
        print_status("Not running inside a virtual environment. We recommend using 'venv\\Scripts\\python.exe'", "WARNING")

def install_dependencies():
    print_status("Checking python dependencies...")
    try:
        import sentence_transformers
        import llama_cpp
        import pytest
        print_status("All core dependencies (sentence-transformers, llama-cpp-python, pytest) are installed.", "SUCCESS")
    except ImportError as e:
        print_status(f"Missing dependency: {e.name}. Installing requirements...", "WARNING")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print_status("Dependencies installed successfully.", "SUCCESS")
        except subprocess.CalledProcessError:
            print_status("Failed to automatically install requirements. Please run 'pip install -r requirements.txt' manually.", "ERROR")
            return False
    return True

def check_models():
    model_dir = Path("models")
    model_path = model_dir / "gemma-2-2b-it-Q4_K_M.gguf"
    
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        
    if model_path.exists():
        size_gb = model_path.stat().st_size / (1024 ** 3)
        print_status(f"Local model found at {model_path} ({size_gb:.2f} GB).", "SUCCESS")
        return True
        
    print_status("Local Gemma 2 2B GGUF model is missing from models/ folder.", "WARNING")
    url = "https://huggingface.co/lmstudio-community/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"
    
    choice = input("Would you like to download the Gemma 2 2B model (~1.6 GB) now? (y/n): ").strip().lower()
    if choice == 'y':
        print_status(f"Downloading model from HuggingFace to {model_path}...")
        print_status("This may take several minutes depending on your internet connection...", "INFO")
        try:
            def report_progress(block_num, block_size, total_size):
                read_so_far = block_num * block_size
                if total_size > 0:
                    percent = read_so_far * 100 / total_size
                    sys.stdout.write(f"\rDownloading: {percent:.1f}% ({read_so_far/(1024**2):.1f}MB / {total_size/(1024**2):.1f}MB)")
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f"\rDownloading: {read_so_far/(1024**2):.1f}MB")
                    sys.stdout.flush()
            
            urllib.request.urlretrieve(url, model_path, reporthook=report_progress)
            print("\n")
            print_status("Model downloaded successfully!", "SUCCESS")
            return True
        except Exception as e:
            print_status(f"\nFailed to download model: {e}", "ERROR")
            print_status("Please manually download the model and place it in the 'models/' folder.", "INFO")
            return False
    else:
        print_status("Skipping model download. Local in-process runs will fall back to the HTTP endpoint.", "INFO")
        return True

def test_local_model_inference():
    """Directly verify that LocalProvider loads and runs the GGUF model.

    This test simulates the EXACT judging environment by setting ALLOWED_MODELS
    to the Fireworks string — the same override the judge injects. It then checks
    that LocalProvider still finds and loads the baked-in GGUF file (not the API).
    """
    print_status("Testing local in-process GGUF inference (simulating ALLOWED_MODELS override)...")

    model_path = Path("models") / "gemma-2-2b-it-Q4_K_M.gguf"
    if not model_path.exists():
        print_status("GGUF model not found at models/gemma-2-2b-it-Q4_K_M.gguf — skipping GGUF test.", "WARNING")
        return False

    try:
        # Temporarily override env to mimic the judging harness
        _orig_allowed = os.environ.get("ALLOWED_MODELS")
        _orig_base_url = os.environ.get("FIREWORKS_BASE_URL")
        _orig_api_key  = os.environ.get("FIREWORKS_API_KEY")

        os.environ["ALLOWED_MODELS"] = "accounts/fireworks/models/gemma2-2b-it,accounts/fireworks/models/gemma2-27b-it"
        os.environ["FIREWORKS_BASE_URL"] = "https://api.fireworks.ai/inference/v1"
        os.environ["FIREWORKS_API_KEY"] = _orig_api_key or "dummy-test-key"

        from config import load_config, ProviderConfig
        from providers import LocalProvider

        # Reload config with the injected env vars — mirrors what the container does on startup
        config = load_config()
        print_status(f"Config loaded: local model string = '{config.local_provider.model}'")

        # Create LocalProvider — it must find the GGUF even with ALLOWED_MODELS overriding config.model
        local = LocalProvider(config.local_provider)

        if local.in_process_model is None:
            print_status("CRITICAL: LocalProvider did NOT load the GGUF model! It fell back to HTTP API.", "ERROR")
            print_status("This means ALL local queries will hit the Fireworks API and will fail during judging.", "ERROR")
            return False

        print_status("LocalProvider loaded the GGUF model successfully.", "SUCCESS")

        # Run a real inference call to verify the model generates output
        import asyncio
        async def _test():
            return await local.generate("What is the capital of France? Reply in one word.")

        result = asyncio.run(_test())

        if result.content and len(result.content.strip()) > 0:
            print_status(f"GGUF inference successful! Response: '{result.content.strip()}'", "SUCCESS")
            return True
        else:
            print_status("GGUF inference returned an empty response.", "ERROR")
            return False

    except Exception as e:
        print_status(f"Local model inference test failed with exception: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Restore original env vars so Phase 1 unit tests run in a clean environment
        for key, orig in [("ALLOWED_MODELS", _orig_allowed), ("FIREWORKS_BASE_URL", _orig_base_url), ("FIREWORKS_API_KEY", _orig_api_key)]:
            if orig is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = orig
        # Evict cached config module so tests re-read a clean environment
        import sys as _sys
        _sys.modules.pop("config", None)


def run_unit_tests():
    print_status("Running unit tests...")
    try:
        result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"], check=True)
        print_status("All unit tests passed!", "SUCCESS")
        return True
    except subprocess.CalledProcessError:
        print_status("Some unit tests failed. Review the pytest output above.", "ERROR")
        return False

def simulate_harness():
    print_status("Simulating evaluation harness task loop...")
    
    input_dir = Path("input")
    output_dir = Path("output")
    tasks_path = input_dir / "tasks.json"
    results_path = output_dir / "results.json"
    
    # Create a dummy task list if not exists
    if not tasks_path.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
        dummy_tasks = [
            {"task_id": "test-factual", "prompt": "What is the boiling point of water?"},
            {"task_id": "test-math", "prompt": "A store sells shirts at $40 with 25% discount. What is the price?"},
            {"task_id": "test-sentiment", "prompt": "Classify the sentiment of this review: The battery life is great, but the screen scratches too easily."}
        ]
        with open(tasks_path, "w", encoding="utf-8") as f:
            json.dump(dummy_tasks, f, indent=2)
        print_status("Created sample tasks at input/tasks.json", "INFO")

    # Clear old outputs
    if results_path.exists():
        results_path.unlink()
        
    # Run the application in harness mode
    print_status("Starting app.py in harness mode...")
    try:
        # Pass absolute paths so app.py detects tasks.json on Windows
        env = os.environ.copy()
        env["TASKS_JSON_PATH"] = str(tasks_path.resolve())
        env["RESULTS_JSON_PATH"] = str(results_path.resolve())
        env["FIREWORKS_API_KEY"] = env.get("FIREWORKS_API_KEY") or "dummy-key-for-local-testing"
        env["FIREWORKS_BASE_URL"] = env.get("FIREWORKS_BASE_URL") or "https://api.fireworks.ai/inference/v1"
        env["ALLOWED_MODELS"] = env.get("ALLOWED_MODELS") or "accounts/fireworks/models/gemma2-2b-it,accounts/fireworks/models/gemma2-27b-it"
        
        # Disable Hugging Face symlinks on Windows to avoid FileNotFoundError
        env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        
        subprocess.run([sys.executable, "app.py"], check=True, env=env)
        
        if results_path.exists():
            print_status("Results file results.json written successfully.", "SUCCESS")
            # Verify output structure
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            
            if isinstance(results, list) and len(results) > 0:
                sample_item = results[0]
                if "task_id" in sample_item and "answer" in sample_item:
                    print_status("results.json matches the REQUIRED evaluation schema exactly!", "SUCCESS")
                    print(json.dumps(results, indent=2))
                    return True
                else:
                    print_status("results.json is missing 'task_id' or 'answer' fields.", "ERROR")
            else:
                print_status("results.json is empty or not a valid list.", "ERROR")
        else:
            print_status("results.json was never written by the application.", "ERROR")
            
    except subprocess.CalledProcessError as e:
        print_status(f"Application crashed during harness simulation: {e}", "ERROR")
        print("\n" + "*" * 60)
        print("DIAGNOSTIC TIP FOR HUGGING FACE CACHE ERROR:")
        print("If you encountered a 'FileNotFoundError' pointing to the huggingface/hub folder,")
        print("it means your local cache for 'all-MiniLM-L6-v2' is corrupted or has symlink issues.")
        print("To fix it, delete this folder and run again:")
        print("  C:\\Users\\sarav\\.cache\\huggingface\\hub\\models--sentence-transformers--all-MiniLM-L6-v2")
        print("*" * 60 + "\n")
    
    return False

def main():
    print("=" * 60)
    print("             AMD AI ROUTER LOCAL VERIFIER")
    print("=" * 60)
    
    check_venv()
    
    if not install_dependencies():
        sys.exit(1)
        
    check_models()

    print("\n--- Phase 0: Local GGUF Model Inference Test ---")
    gguf_ok = test_local_model_inference()

    print("\n--- Phase 1: Unit Tests ---")
    tests_ok = run_unit_tests()
    
    print("\n--- Phase 2: Harness Simulation ---")
    harness_ok = simulate_harness()
    
    print("\n" + "=" * 60)
    if gguf_ok and tests_ok and harness_ok:
        print("\033[92mCONGRATULATIONS! Your local router setup is fully verified and clean.\033[0m")
        print("You are ready to build the Docker container and push to Docker Hub.")
        print("\nBuild command:")
        print("  docker build --platform linux/amd64 -t your_username/amd-ai-router:latest .")
    else:
        if not gguf_ok:
            print("\033[91mCRITICAL: Local GGUF model inference FAILED. Do NOT build the Docker image yet.\033[0m")
            print("Fix the providers.py GGUF path resolution before proceeding.")
        else:
            print("\033[91mVERIFICATION FAILED. Please review the errors above and fix them before building.\033[0m")
    print("=" * 60)

if __name__ == "__main__":
    main()
