"""Router module — the brain of the project.

Analyzes requests and decides whether to use local or remote inference,
handling automatic fallback on failure.

Routing pipeline:
    User Prompt → semantic kNN classifier → score + category → threshold → Provider → Response

Routing Strategies (selectable via config.yaml):
    semantic    — vector similarity kNN classifier using sentence-transformers (ACTIVE DEFAULT)
    heuristic   — keyword + length scoring (fallback when model unavailable)
    gatekeeper  — tiny local model classifier (stub)

The semantic strategy embeds the incoming prompt and compares it against 40
hand-curated reference examples (5 local + 5 remote per hackathon category)
using cosine similarity kNN voting. The model runs in-process on CPU with
zero Fireworks API calls — costs 0 tokens under the hackathon's scoring rules.

Hackathon categories covered:
    1. Factual knowledge          → local  (simple recall)
    2. Mathematical reasoning     → remote (multi-step arithmetic)
    3. Sentiment classification   → local  (straightforward labelling)
    4. Text summarisation         → local  (condensing is easy for 2B models)
    5. Named entity recognition   → local  (structured extraction)
    6. Code debugging             → remote (bug analysis requires deep reasoning)
    7. Logical / deductive reason → remote (constraint satisfaction is hard)
    8. Code generation            → remote (correctness matters most)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from providers import Provider, ProviderResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing Strategy Enum
# ---------------------------------------------------------------------------

class RoutingStrategy(str, Enum):
    """Available routing strategies."""
    HEURISTIC = "heuristic"
    SEMANTIC = "semantic"       # active — sentence-transformers kNN
    GATEKEEPER = "gatekeeper"   # stub — Phase 7


# ---------------------------------------------------------------------------
# Heuristic signals — tuned from benchmark results
# ---------------------------------------------------------------------------

# Code patterns: markdown fences, python keywords, common operators
_CODE_PATTERNS = re.compile(
    r"```|"
    r"\bdef\b|\bclass\b|\bimport\b|\bfunction\b|\breturn\b|"
    r"\bfor\b.*\brange\b|\bwhile\b|\blambda\b|"
    r"=>|\{\s*\}|\(\s*\)|;$",
    re.MULTILINE,
)

# Math patterns: symbols, equations, calculation keywords
_MATH_PATTERNS = re.compile(
    r"\b(calculate|solve|compute|integral|derivative|equation|"
    r"matrix|vector|probability|factorial|theorem|proof|"
    r"sum|product|series|limit|modulo|prime|divisible)\b|"
    r"[∫∑∏√±×÷≠≤≥∞]|"
    r"\d+\s*[\+\-\*/\^]\s*\d+",
    re.IGNORECASE,
)

# Reasoning / analytical keywords
_REASONING_PATTERNS = re.compile(
    r"\b(explain|compare|analyze|evaluate|critique|describe|"
    r"summarize|design|architect|strategy|difference between|"
    r"pros and cons|step[- ]by[- ]step|in detail|trade.?off)\b",
    re.IGNORECASE,
)

# Category detection patterns (aligns with the 8 Hackathon categories)
_MATH_CATEGORY = re.compile(
    r"\b(math|mathematics|calculate|calculation|solve|solving|solver|equation|integral|integration|derivative|derivation|matrix|probability|"
    r"factorial|prime|formula|theorem|geometry|geometric|algebra|algebraic|arithmetic)\b|"
    r"[∫∑∏√±×÷≠≤≥]|\d+\s*[\+\-\*/\^]\s*\d+",
    re.IGNORECASE,
)
_CODE_CATEGORY = re.compile(
    r"\b(function|class|program|code|algorithm|implement|implementation|"
    r"script|debug|debugging|recursion|loop|array|string|bug|fix)\b|```",
    re.IGNORECASE,
)
_CLASSIFICATION_CATEGORY = re.compile(
    r"\b(classify|categorize|label|sentiment|is this|which type|"
    r"identify whether|detect)\b",
    re.IGNORECASE,
)
_SUMMARIZATION_CATEGORY = re.compile(
    r"\b(summarize|summarizing|summary|brief|overview|in one sentence|"
    r"tldr|key points|main idea)\b",
    re.IGNORECASE,
)
_NER_CATEGORY = re.compile(
    r"\b(extract|named entities|named entity|ner|entities|location|date|organization|person)\b",
    re.IGNORECASE,
)
_REASONING_CATEGORY = re.compile(
    r"\b(logic|logical|deductive|puzzle|riddle|constraint|friend|pet|who owns|statement)\b",
    re.IGNORECASE,
)

# Prompt length thresholds (characters)
_SHORT_PROMPT = 120
_LONG_PROMPT = 500


# ---------------------------------------------------------------------------
# Feature extraction + category detection
# ---------------------------------------------------------------------------


def analyze_prompt(prompt: str) -> tuple[float, str, str]:
    """Extract features and return (complexity_score, reason, category).

    The complexity score is in [0.0, 1.0]. Higher → more complex → prefer remote.

    Scoring (additive, capped at 1.0):
        Length ≥ 500 chars:             +0.3
        Length > 120 chars:             +0.1
        Code patterns detected:         +0.4
        Math patterns detected:         +0.3
        Reasoning/analysis keywords:    +0.2

    Category detection (used for per-category threshold lookup):
        mathematics, programming, classification, summarization, ner, logical_reasoning, general

    Args:
        prompt: The user's input text.

    Returns:
        Tuple of (score: float, reason: str, category: str).
    """
    score = 0.0
    signals: list[str] = []

    # --- Length scoring ---
    length = len(prompt)
    if length >= _LONG_PROMPT:
        score += 0.3
        signals.append(f"long prompt ({length} chars)")
    elif length > _SHORT_PROMPT:
        score += 0.1
        signals.append(f"medium prompt ({length} chars)")

    # --- Signal scoring ---
    has_code = bool(_CODE_PATTERNS.search(prompt))
    has_math = bool(_MATH_PATTERNS.search(prompt))
    has_reasoning = bool(_REASONING_PATTERNS.search(prompt))

    if has_code:
        score += 0.4
        signals.append("code detected")

    if has_math:
        score += 0.3
        signals.append("math detected")

    if has_reasoning:
        score += 0.2
        signals.append("reasoning/analysis keywords")

    score = min(score, 1.0)
    reason = ", ".join(signals) if signals else "simple prompt"

    # --- Category detection (ordered by specificity) ---
    if _CODE_CATEGORY.search(prompt):
        category = "programming"
    elif _MATH_CATEGORY.search(prompt):
        category = "mathematics"
    elif _REASONING_CATEGORY.search(prompt):
        category = "logical_reasoning"
    elif _NER_CATEGORY.search(prompt):
        category = "ner"
    elif _CLASSIFICATION_CATEGORY.search(prompt):
        category = "classification"
    elif _SUMMARIZATION_CATEGORY.search(prompt):
        category = "summarization"
    else:
        category = "general"

    return score, reason, category


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RoutingResult:
    """Complete record of a routing decision — used for benchmarking."""

    response: ProviderResponse
    provider_used: str
    model_used: str
    routing_reason: str
    complexity_score: float
    category: str = "general"
    threshold_used: float = 0.5
    fallback_used: bool = False
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Semantic Similarity Routing (Strategy: semantic)
# ---------------------------------------------------------------------------

# The embedding model is pre-loaded at container startup via preload_semantic_model().
# Falls back to lazy-loading on first _semantic_score() call if not pre-loaded.
# Uses all-MiniLM-L6-v2 (~100 MB, CPU-only, no GPU needed).
_semantic_model = None
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Reference corpus — 40 examples, 5 local + 5 remote per hackathon category.
#
# label 0 → route LOCAL  (small Gemma 2B handles this well, costs 0 tokens)
# label 1 → route REMOTE (needs the better model on Fireworks)
#
# Design rules:
#   • Use phrasing identical to real evaluation prompts (see practice tasks)
#   • Each category has equal representation (balanced corpus)
#   • Local examples are genuinely simple; remote are multi-step / complex
# ---------------------------------------------------------------------------
_REFERENCE_PROMPTS: list[tuple[str, int]] = [

    # ── 1. FACTUAL KNOWLEDGE ── (local: recall, definitions, simple facts)
    ("What is the capital of Australia and what is it near?", 0),
    ("What does RAM stand for in computing?", 0),
    ("Who invented the telephone?", 0),
    ("What is the boiling point of water at sea level?", 0),
    ("What year did the Berlin Wall fall?", 0),
    # factual but complex / multi-step explanation → remote
    ("Explain in detail how TCP/IP handshaking works and why each step is necessary.", 1),
    ("Describe the full lifecycle of a star from nebula to white dwarf or supernova.", 1),
    ("What are the causes and long-term consequences of the 2008 global financial crisis?", 1),
    ("Explain how CRISPR-Cas9 gene editing works at the molecular level.", 1),
    ("Compare the philosophical differences between utilitarianism and Kantian ethics.", 1),

    # ── 2. MATHEMATICAL REASONING ── (local: trivial arithmetic; remote: multi-step)
    ("What is 15% of 200?", 0),
    ("If a shirt costs $40 and is 25% off, what is the final price?", 0),
    ("What is 7 squared?", 0),
    ("A rectangle is 8 cm wide and 5 cm tall. What is its area?", 0),
    ("Convert 0.75 to a percentage.", 0),
    # multi-step word problems → remote
    ("A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many remain?", 1),
    ("Solve: if compound interest of 5% per year is applied to $1000, what is the value after 3 years?", 1),
    ("A train travels 300 km at 80 km/h, then 150 km at 60 km/h. What is the average speed for the full journey?", 1),
    ("Solve the system of equations: 2x + 3y = 12 and x - y = 1.", 1),
    ("A population grows at 3% per year. Starting at 500,000 how many years to reach 700,000?", 1),

    # ── 3. SENTIMENT CLASSIFICATION ── (all local: Gemma 2B handles this well)
    ("Classify the sentiment of this review: The battery life is great, but the screen scratches too easily.", 0),
    ("Is this statement positive, negative, or neutral: Shipping was fast but the item arrived damaged.", 0),
    ("Label the sentiment: I absolutely love this product, it exceeded all my expectations!", 0),
    ("What is the sentiment of: The movie was okay, nothing special but not terrible either.", 0),
    ("Classify: Worst customer service I have ever experienced. Will never buy again.", 0),
    # complex multi-aspect sentiment analysis → remote
    ("Analyse the overall sentiment and per-aspect sentiment for this long product review covering price, quality, delivery, and support.", 1),
    ("Identify all sentiment shifts in this paragraph and explain the emotional arc of the text.", 1),
    ("Compare the sentiment intensity between these five customer reviews and rank them from most negative to most positive with justification.", 1),
    ("Extract fine-grained opinion targets and their associated sentiments from this restaurant review.", 1),
    ("Perform aspect-based sentiment analysis on this hotel review covering cleanliness, staff, location, and value.", 1),

    # ── 4. TEXT SUMMARISATION ── (local: short summaries; remote: long / constrained)
    ("Summarize the following in exactly one sentence: The sky is blue because of Rayleigh scattering of sunlight.", 0),
    ("Give a one-sentence overview of what photosynthesis is.", 0),
    ("What are the key points of this text: Python is a high-level, interpreted programming language.", 0),
    ("Condense this into a brief summary: Water covers 71% of the Earth's surface and is essential for all known life.", 0),
    ("TL;DR of: The Eiffel Tower was built in 1889 as the entrance arch for the 1889 World's Fair.", 0),
    # long-form or heavily constrained summarisation → remote
    ("Summarize this 500-word technical document into exactly three bullet points, each under 15 words, preserving all key metrics.", 1),
    ("Extract and summarize the main argument, evidence, and conclusion of this academic paper abstract.", 1),
    ("Produce an executive summary of this earnings report in two paragraphs, highlighting revenue, costs, and year-over-year changes.", 1),
    ("Summarize this legal contract section identifying the key obligations, deadlines, and penalties for breach.", 1),
    ("Create a structured summary with headings for each major topic covered in this 800-word article.", 1),

    # ── 5. NAMED ENTITY RECOGNITION ── (local: simple lists; remote: nested/complex)
    ("Extract all named entities from: Maria Sanchez joined Fireworks AI in Berlin last March.", 0),
    ("Find all person names in: John Smith met with Alice Johnson at the conference.", 0),
    ("List the organisations mentioned in: Apple and Microsoft announced a partnership in Seattle.", 0),
    ("What locations are mentioned in: The flight from Paris to Tokyo landed in two hours.", 0),
    ("Extract dates from: The meeting is on 14 July 2025 and the deadline is 31 August 2025.", 0),
    # complex NER with nested / ambiguous entities → remote
    ("Extract and classify all entities (person, org, location, date, product) from this three-paragraph news article.", 1),
    ("Identify all named entities, resolve coreferences, and link each entity to its canonical form in the text.", 1),
    ("Extract relationships between entities: who works where, who met whom, and when in this document.", 1),
    ("Identify all ambiguous entity mentions and explain why each could belong to multiple categories.", 1),
    ("Extract entities and build a knowledge graph of relationships from this company announcement.", 1),

    # ── 6. CODE DEBUGGING ── (local: obvious bugs; remote: subtle / multi-bug)
    ("This function should return the max of a list but has a bug: def get_max(nums): return nums[0]. Find and fix it.", 0),
    ("Find the bug: def add(a, b): return a - b", 0),
    ("What is wrong with: for i in range(10): print(i) if i == 5 break", 0),
    ("Fix the off-by-one error in: for i in range(1, len(arr)): print(arr[i])", 0),
    ("Identify the syntax error: def greet(name) print('Hello', name)", 0),
    # subtle / multi-file / algorithmic bugs → remote
    ("This binary search implementation returns wrong results for edge cases. Find all bugs and provide a corrected version with explanation.", 1),
    ("Debug this async Python code that causes a race condition under concurrent load and explain the fix.", 1),
    ("Identify the memory leak in this C++ class, explain why it occurs, and provide a safe implementation.", 1),
    ("This SQL query returns duplicate rows in some cases. Analyse the joins and fix the logic.", 1),
    ("Find all bugs in this recursive tree traversal and explain the time complexity of both the buggy and fixed versions.", 1),

    # ── 7. LOGICAL / DEDUCTIVE REASONING ── (local: trivial; remote: multi-constraint)
    ("All cats are animals. Whiskers is a cat. Is Whiskers an animal?", 0),
    ("If today is Monday, what day is it in 3 days?", 0),
    ("A is taller than B. B is taller than C. Who is the shortest?", 0),
    ("There are 5 apples. I eat 2. How many are left?", 0),
    ("It always rains when it is cloudy. It is cloudy. Will it rain?", 0),
    # multi-constraint puzzles → remote
    ("Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. Sam does not own the bird. Jo owns the dog. Who owns the cat?", 1),
    ("Five houses in a row are each a different colour. The Englishman lives in the red house. Solve for which nationality lives in the green house.", 1),
    ("Four people need to cross a bridge at night with one torch. The bridge holds two at a time. Minimum time crossing given their speeds.", 1),
    ("Given these six clues about six people's occupations and hobbies, determine who is the photographer.", 1),
    ("Determine the truth-teller and liar from: A says 'B always lies', B says 'C sometimes lies', C says 'A always tells truth'.", 1),

    # ── 8. CODE GENERATION ── (local: trivial one-liners; remote: spec-complete functions)
    ("Write a Python function that adds two numbers.", 0),
    ("Write a function that returns True if a number is even.", 0),
    ("Write a Python function that reverses a string.", 0),
    ("Write a function that returns the length of a list.", 0),
    ("Write a Python function that checks if a string is a palindrome.", 0),
    # complex spec-complete / edge-case-aware functions → remote
    ("Write a Python function that returns the second-largest number in a list, handling duplicates correctly.", 1),
    ("Implement a thread-safe LRU cache in Python with O(1) get and put operations.", 1),
    ("Write a Python class implementing a binary search tree with insert, delete, search, and in-order traversal.", 1),
    ("Implement Dijkstra's shortest-path algorithm in Python returning both distances and the path.", 1),
    ("Write a Python decorator that retries a function up to N times with exponential backoff on exception.", 1),
]

_reference_texts: list[str] = [p for p, _ in _REFERENCE_PROMPTS]
_reference_labels: list[int] = [lbl for _, lbl in _REFERENCE_PROMPTS]
_reference_embeddings = None   # numpy array, computed once at load time


def preload_semantic_model() -> bool:
    """Eagerly load and index the embedding model at container startup.

    Call this once during app initialisation (before serving requests) so that
    the 60-second startup deadline is met and the first request is not slow.

    Returns True if the model loaded successfully, False otherwise.
    """
    model = _get_semantic_model()
    if model is not None:
        logger.info("Semantic router ready — %d reference prompts indexed.", len(_reference_texts))
        return True
    logger.warning("Semantic router unavailable — heuristic fallback will be used.")
    return False


def _get_semantic_model():
    """Load the embedding model and pre-compute reference embeddings (idempotent)."""
    global _semantic_model, _reference_embeddings
    if _semantic_model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            logger.info("Loading semantic embedding model '%s'...", _EMBED_MODEL_NAME)
            _semantic_model = SentenceTransformer(_EMBED_MODEL_NAME)
            _reference_embeddings = _semantic_model.encode(
                _reference_texts, convert_to_numpy=True, normalize_embeddings=True
            )
            logger.info(
                "Semantic model loaded — %d reference prompts indexed.",
                len(_reference_texts),
            )
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
            return None
        except Exception as exc:
            logger.error("Failed to load semantic model: %s", exc)
            return None
    return _semantic_model


def _semantic_score(prompt: str, k: int = 7) -> tuple[float, str]:
    """Score a prompt using cosine-similarity kNN against the reference corpus.

    Args:
        prompt: Incoming user prompt.
        k:      Number of nearest neighbours to vote on routing (default 7,
                odd number avoids ties, covers at least 1 example per category).

    Returns:
        (score, reason) where score ∈ [0.0, 1.0].
        score = fraction of k neighbours labelled 'remote'.
        0.0 → all neighbours local  → strong local signal
        1.0 → all neighbours remote → strong remote signal
    """
    import numpy as np  # type: ignore
    model = _get_semantic_model()
    if model is None or _reference_embeddings is None:
        return 0.5, "semantic model unavailable (fallback score 0.5)"

    prompt_emb = model.encode([prompt], convert_to_numpy=True, normalize_embeddings=True)
    # Dot product = cosine similarity (embeddings are L2-normalised)
    similarities = (_reference_embeddings @ prompt_emb.T).flatten()
    top_k_idx = similarities.argsort()[-k:][::-1]
    remote_votes = sum(_reference_labels[i] for i in top_k_idx)
    score = remote_votes / k
    nearest = _reference_texts[top_k_idx[0]]
    nearest_display = nearest[:50] + "..." if len(nearest) > 50 else nearest
    reason = (
        f"semantic kNN k={k} votes={remote_votes}/{k} "
        f"category={'remote' if score > 0.5 else 'local'} "
        f"near='{nearest_display}'"
    )
    return score, reason


async def _semantic_route(
    prompt: str,
    local_provider: Provider,
    remote_provider: Provider,
    threshold: float = 0.5,
    fallback_enabled: bool = True,
    category_thresholds: dict[str, float] | None = None,
) -> RoutingResult:
    """Semantic vector-similarity routing using sentence-transformers.

    Embeds the incoming prompt and compares it against labelled reference
    examples (local vs. remote) using cosine similarity kNN voting.
    The fraction of remote-labelled neighbours becomes the complexity score.

    Falls back to heuristic routing if the model is unavailable.
    """
    score, reason = _semantic_score(prompt)
    if "unavailable" in reason:
        logger.warning("Semantic model unavailable — falling back to heuristic routing.")
        return await _heuristic_route(
            prompt, local_provider, remote_provider,
            threshold=threshold,
            fallback_enabled=fallback_enabled,
            category_thresholds=category_thresholds,
        )

    # Category detection still runs so per-category thresholds apply
    _, _, category = analyze_prompt(prompt)
    effective_threshold = threshold
    if category_thresholds and category in category_thresholds:
        effective_threshold = category_thresholds[category]

    logger.info(
        "Semantic routing — score=%.2f threshold=%.2f category=%s",
        score, effective_threshold, category,
    )

    if score > effective_threshold:
        logger.info("Semantic → REMOTE (score %.2f > %.2f)", score, effective_threshold)
        response = await remote_provider.generate(prompt)
        return RoutingResult(
            response=response,
            provider_used="remote",
            model_used=response.model,
            routing_reason=reason,
            complexity_score=score,
            category=category,
            threshold_used=effective_threshold,
            fallback_used=False,
        )

    logger.info("Semantic → LOCAL (score %.2f ≤ %.2f)", score, effective_threshold)
    try:
        response = await local_provider.generate(prompt)
        return RoutingResult(
            response=response,
            provider_used="local",
            model_used=response.model,
            routing_reason=reason,
            complexity_score=score,
            category=category,
            threshold_used=effective_threshold,
            fallback_used=False,
        )
    except Exception as local_error:
        if not fallback_enabled:
            raise
        logger.warning("Local failed in semantic mode (%s) — escalating to remote.", local_error)
        response = await remote_provider.generate(prompt)
        return RoutingResult(
            response=response,
            provider_used="remote",
            model_used=response.model,
            routing_reason=f"fallback (local failed: {local_error})",
            complexity_score=score,
            category=category,
            threshold_used=effective_threshold,
            fallback_used=True,
        )


# ---------------------------------------------------------------------------
# Gatekeeper Model Routing (Strategy: gatekeeper) — stub
# ---------------------------------------------------------------------------


async def _gatekeeper_route(
    prompt: str,
    local_provider: Provider,
    remote_provider: Provider,
    **kwargs,
) -> RoutingResult:
    """Gatekeeper model routing — stub (Phase 7+).

    When implemented, this strategy will:
    1. Send the prompt to a tiny (1B parameter) local classifier model via
       the local_provider using a structured classification prompt.
    2. Ask it to rate difficulty 1–5 in a single token.
    3. Use the rating to drive the routing decision.

    Falls back to heuristic strategy until implemented.
    """
    logger.warning(
        "Gatekeeper strategy is not yet implemented. Falling back to heuristic routing."
    )
    return await _heuristic_route(prompt, local_provider, remote_provider, **kwargs)


# ---------------------------------------------------------------------------
# Heuristic routing (active strategy)
# ---------------------------------------------------------------------------


async def _heuristic_route(
    prompt: str,
    local_provider: Provider,
    remote_provider: Provider,
    threshold: float = 0.5,
    fallback_enabled: bool = True,
    category_thresholds: dict[str, float] | None = None,
) -> RoutingResult:
    """Core heuristic routing logic."""
    score, reason, category = analyze_prompt(prompt)

    # Category-specific threshold overrides the global threshold
    effective_threshold = threshold
    if category_thresholds and category in category_thresholds:
        effective_threshold = category_thresholds[category]
        logger.info(
            "Category '%s' threshold override: %.2f (global: %.2f)",
            category, effective_threshold, threshold,
        )

    logger.info(
        "Routing decision — score=%.2f threshold=%.2f category=%s signals=[%s]",
        score, effective_threshold, category, reason,
    )

    # --- Remote path (complex request) ---
    if score > effective_threshold:
        logger.info(
            "Routing to REMOTE (score %.2f > threshold %.2f, category=%s)",
            score, effective_threshold, category,
        )
        response = await remote_provider.generate(prompt)
        return RoutingResult(
            response=response,
            provider_used="remote",
            model_used=response.model,
            routing_reason=reason,
            complexity_score=score,
            category=category,
            threshold_used=effective_threshold,
            fallback_used=False,
        )

    # --- Local path (simple request) ---
    logger.info(
        "Routing to LOCAL (score %.2f ≤ threshold %.2f, category=%s)",
        score, effective_threshold, category,
    )
    try:
        response = await local_provider.generate(prompt)
        return RoutingResult(
            response=response,
            provider_used="local",
            model_used=response.model,
            routing_reason=reason,
            complexity_score=score,
            category=category,
            threshold_used=effective_threshold,
            fallback_used=False,
        )
    except Exception as local_error:
        if not fallback_enabled:
            logger.error(
                "Local provider failed and fallback is disabled: %s", local_error
            )
            raise

        logger.warning(
            "Local provider failed (%s) — falling back to REMOTE", local_error
        )
        response = await remote_provider.generate(prompt)
        return RoutingResult(
            response=response,
            provider_used="remote",
            model_used=response.model,
            routing_reason=f"fallback (local failed: {local_error})",
            complexity_score=score,
            category=category,
            threshold_used=effective_threshold,
            fallback_used=True,
        )


# ---------------------------------------------------------------------------
# Public route() dispatcher
# ---------------------------------------------------------------------------


async def route(
    prompt: str,
    local_provider: Provider,
    remote_provider: Provider,
    threshold: float = 0.5,
    fallback_enabled: bool = True,
    category_thresholds: dict[str, float] | None = None,
    strategy: str = "heuristic",
) -> RoutingResult:
    """Route a prompt to the cheapest provider that can answer it correctly.

    Selects the routing strategy from config and dispatches accordingly.

    Args:
        prompt: The user's input prompt.
        local_provider: Local inference provider (cheap, fast).
        remote_provider: Remote inference provider (accurate, costly).
        threshold: Global complexity score threshold.
        fallback_enabled: If True, failed local calls escalate to remote.
        category_thresholds: Per-category threshold overrides.
        strategy: Routing strategy to use ('heuristic', 'semantic', 'gatekeeper').

    Returns:
        A RoutingResult with the response and full routing metadata.

    Raises:
        Exception: If the provider also fails (no further fallback).
    """
    kwargs = {
        "threshold": threshold,
        "fallback_enabled": fallback_enabled,
        "category_thresholds": category_thresholds or {},
    }

    try:
        routing_strategy = RoutingStrategy(strategy)
    except ValueError:
        logger.warning("Unknown strategy '%s', defaulting to heuristic.", strategy)
        routing_strategy = RoutingStrategy.HEURISTIC

    if routing_strategy == RoutingStrategy.SEMANTIC:
        return await _semantic_route(prompt, local_provider, remote_provider, **kwargs)
    elif routing_strategy == RoutingStrategy.GATEKEEPER:
        return await _gatekeeper_route(prompt, local_provider, remote_provider, **kwargs)
    else:
        return await _heuristic_route(prompt, local_provider, remote_provider, **kwargs)
