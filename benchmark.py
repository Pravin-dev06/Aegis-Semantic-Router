"""Benchmark module — measures routing quality.

Loads evaluation datasets, runs prompts through the router asynchronously
with concurrency limits, evaluates accuracy and cost metrics, and prints
a comprehensive evaluation report.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from providers import Provider
from router import route

logger = logging.getLogger(__name__)


@dataclass
class TestCaseResult:
    """Detailed evaluation result for a single benchmark test case."""

    case_id: int
    prompt: str
    expected: str
    category: str
    is_correct: bool
    provider_used: str
    model_used: str
    latency_sec: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    complexity_score: float
    fallback_used: bool


@dataclass
class BenchmarkReport:
    """Compiled metrics and statistics for the benchmark run."""

    timestamp: str
    local_model: str
    remote_model: str
    threshold: float
    total_cases: int
    correct_cases: int
    accuracy: float
    total_remote_tokens: int
    total_remote_prompt_tokens: int
    total_remote_completion_tokens: int
    local_routing_count: int
    remote_routing_count: int
    local_routing_pct: float
    remote_routing_pct: float
    fallback_count: int
    avg_latency_sec: float
    results: list[TestCaseResult]


def evaluate_response(response_content: str, expected_content: str) -> bool:
    """Evaluate response correctness using a substring match.

    Args:
        response_content: The output content from the model.
        expected_content: The ground truth target string.

    Returns:
        True if the expected substring is present in the response (case-insensitive).
    """
    clean_response = response_content.strip().lower()
    clean_expected = expected_content.strip().lower()
    return clean_expected in clean_response


async def run_benchmark(
    dataset_path: str,
    local_provider: Provider,
    remote_provider: Provider,
    threshold: float = 0.5,
    fallback_enabled: bool = True,
    concurrency_limit: int = 5,
    output_dir: str = "reports",
    category_thresholds: dict[str, float] | None = None,
    strategy: str = "heuristic",
) -> BenchmarkReport:
    """Run the benchmark evaluation suite.

    Args:
        dataset_path: Path to the JSON dataset file.
        local_provider: Local LLM provider instance.
        remote_provider: Remote LLM provider instance.
        threshold: Router complexity threshold.
        fallback_enabled: Whether to escalate local failures to remote.
        concurrency_limit: Maximum concurrent provider execution tasks.
        output_dir: Directory where the run report JSON will be saved.
        category_thresholds: Category-specific threshold overrides.
        strategy: Routing strategy to use.

    Returns:
        A BenchmarkReport with metrics and individual test results.
    """
    logger.info("Starting benchmark run for dataset: %s", dataset_path)
    
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    if not isinstance(dataset, list):
        raise ValueError("Dataset file must be a JSON list of test cases")

    semaphore = asyncio.Semaphore(concurrency_limit)
    results: list[TestCaseResult] = []

    async def _evaluate_case(case: dict) -> None:
        case_id = case.get("id", -1)
        prompt = case.get("prompt", "")
        expected = case.get("expected", "")
        category = case.get("category", "general")

        async with semaphore:
            start_time = time.perf_counter()
            try:
                routing_result = await route(
                    prompt=prompt,
                    local_provider=local_provider,
                    remote_provider=remote_provider,
                    threshold=threshold,
                    fallback_enabled=fallback_enabled,
                    category_thresholds=category_thresholds,
                    strategy=strategy,
                )
                latency = time.perf_counter() - start_time
                is_correct = evaluate_response(routing_result.response.content, expected)
                
                results.append(
                    TestCaseResult(
                        case_id=case_id,
                        prompt=prompt,
                        expected=expected,
                        category=category,
                        is_correct=is_correct,
                        provider_used=routing_result.provider_used,
                        model_used=routing_result.model_used,
                        latency_sec=latency,
                        prompt_tokens=routing_result.response.prompt_tokens,
                        completion_tokens=routing_result.response.completion_tokens,
                        total_tokens=routing_result.response.prompt_tokens + routing_result.response.completion_tokens,
                        complexity_score=routing_result.complexity_score,
                        fallback_used=routing_result.fallback_used,
                    )
                )
            except Exception as e:
                latency = time.perf_counter() - start_time
                logger.error("Failed to evaluate test case %d: %s", case_id, e)
                results.append(
                    TestCaseResult(
                        case_id=case_id,
                        prompt=prompt,
                        expected=expected,
                        category=category,
                        is_correct=False,
                        provider_used="failed",
                        model_used="failed",
                        latency_sec=latency,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        complexity_score=0.0,
                        fallback_used=False,
                    )
                )

    # Execute all test cases concurrently with throttle limit
    await asyncio.gather(*[_evaluate_case(case) for case in dataset])

    # Sort results back to original dataset order
    results.sort(key=lambda r: r.case_id)

    # Compile report metrics
    total_cases = len(results)
    correct_cases = sum(1 for r in results if r.is_correct)
    accuracy = (correct_cases / total_cases) if total_cases > 0 else 0.0

    # Calculate token counts — only count remote usage
    remote_results = [r for r in results if r.provider_used == "remote"]
    total_remote_prompt = sum(r.prompt_tokens for r in remote_results)
    total_remote_completion = sum(r.completion_tokens for r in remote_results)
    total_remote_tokens = total_remote_prompt + total_remote_completion

    local_count = sum(1 for r in results if r.provider_used == "local")
    remote_count = sum(1 for r in results if r.provider_used == "remote")
    local_pct = (local_count / total_cases * 100.0) if total_cases > 0 else 0.0
    remote_pct = (remote_count / total_cases * 100.0) if total_cases > 0 else 0.0

    fallback_count = sum(1 for r in results if r.fallback_used)
    avg_latency = (sum(r.latency_sec for r in results) / total_cases) if total_cases > 0 else 0.0

    report = BenchmarkReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        local_model=getattr(local_provider.config, "model", "unknown"),
        remote_model=getattr(remote_provider.config, "model", "unknown"),
        threshold=threshold,
        total_cases=total_cases,
        correct_cases=correct_cases,
        accuracy=accuracy,
        total_remote_tokens=total_remote_tokens,
        total_remote_prompt_tokens=total_remote_prompt,
        total_remote_completion_tokens=total_remote_completion,
        local_routing_count=local_count,
        remote_routing_count=remote_count,
        local_routing_pct=local_pct,
        remote_routing_pct=remote_pct,
        fallback_count=fallback_count,
        avg_latency_sec=avg_latency,
        results=results,
    )

    # Save JSON report
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(output_dir) / f"benchmark_{timestamp_str}.json"
        with open(report_file, "w", encoding="utf-8") as rf:
            json.dump(asdict(report), rf, indent=2)
        logger.info("Benchmark report saved to: %s", report_file)
    except Exception as save_err:
        logger.error("Failed to save benchmark JSON report: %s", save_err)

    # Display clean table report in console
    print_report_table(report)

    return report


def print_report_table(report: BenchmarkReport) -> None:
    """Print a clean, visually structured report to stdout."""
    print("\n" + "=" * 65)
    print("                      BENCHMARK REPORT")
    print("=" * 65)
    print(f"Timestamp:       {report.timestamp}")
    print(f"Local Model:     {report.local_model}")
    print(f"Remote Model:    {report.remote_model}")
    print(f"Threshold:       {report.threshold:.2f}")
    print("-" * 65)
    print(f"Total Cases:     {report.total_cases:<10} | Accuracy:       {report.accuracy * 100.0:.1f}%")
    print(f"Local Routed:    {report.local_routing_count} ({report.local_routing_pct:.1f}%) | Remote Routed:  {report.remote_routing_count} ({report.remote_routing_pct:.1f}%)")
    print(f"Fallbacks Used:  {report.fallback_count:<10} | Avg Latency:    {report.avg_latency_sec:.3f}s")
    print("-" * 65)
    print("REMOTE TOKENS CONSUMED:")
    print(f"  Prompt:        {report.total_remote_prompt_tokens}")
    print(f"  Completion:    {report.total_remote_completion_tokens}")
    print(f"  Total:         {report.total_remote_tokens}")
    print("=" * 65)
    
    print("\nCase Results Summary:")
    print(f"{'ID':<4} | {'Category':<15} | {'Decision':<8} | {'Score':<5} | {'Correct':<7} | {'Latency':<7}")
    print("-" * 65)
    for res in report.results:
        correct_str = "PASS" if res.is_correct else "FAIL"
        print(f"{res.case_id:<4} | {res.category:<15} | {res.provider_used:<8} | {res.complexity_score:<5.2f} | {correct_str:<7} | {res.latency_sec:<6.3f}s")
    print("=" * 65 + "\n")
