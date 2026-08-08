"""Utilities for producing content-domain Hugging Face search queries."""

from __future__ import annotations

import re


_GENERIC_QUERY_TOKENS = {
    "huggingface",
    "hugging",
    "face",
    "hf",
    "dataset",
    "datasets",
    "data",
    "search",
    "training",
    "train",
    "task",
    "tasks",
    "improve",
    "enhance",
    "increase",
    "ability",
    "abilities",
    "skill",
    "skills",
    "problem",
    "problems",
    "solving",
    "for",
    "with",
    "to",
    "of",
    "in",
    "the",
    "and",
}

_DIFFICULTY_QUERY_TOKENS = {
    "easy",
    "medium",
    "hard",
    "difficulty",
    "difficulties",
    "beginner",
    "intermediate",
    "advanced",
}


_DOMAIN_FALLBACKS = [
    (("代码", "编程", "软件", "程序", "code", "coding", "programming", "software", "github", "patch", "bug"), "code repair"),
    (("数学", "math", "arithmetic", "algebra", "geometry"), "math reasoning"),
    (("翻译", "translation", "translate"), "translation"),
    (("推理", "reasoning", "logic"), "reasoning"),
]


def fallback_query_from_goal(goal: str) -> str:
    """Map a user goal to a safe content-domain query.

    The returned phrase describes the dataset content, not the search platform.
    """
    text = (goal or "").strip().lower()
    for needles, query in _DOMAIN_FALLBACKS:
        if any(needle in text for needle in needles):
            return query
    return "reasoning instruction"


def _content_query_tokens(raw_query: object) -> list[str]:
    query = str(raw_query or "").strip().lower()
    query = re.sub(r"[^a-z0-9+.#\-\s]", " ", query)
    return [token for token in query.split() if token not in _GENERIC_QUERY_TOKENS]


def is_difficulty_only_search_query(raw_query: object) -> bool:
    """Return true when a query contains only difficulty metadata tokens."""
    tokens = _content_query_tokens(raw_query)
    return bool(tokens) and all(token in _DIFFICULTY_QUERY_TOKENS for token in tokens)


def normalize_content_search_query(raw_query: object, goal: str = "", fallback: str = "") -> str:
    """Remove tool/platform words and return a short content-domain query.

    Small local models sometimes output strings like ``HuggingFace dataset search``
    because the prompt mentions the search tool.  Such strings are not useful HF
    queries, so this function strips platform/generic words and falls back to the
    user's domain goal when the remaining content is empty.

    Difficulty-only strings (``medium difficulty``, ``easy``) are metadata, not
    content queries.  The normalizer deliberately drops them instead of turning
    them into HF keywords; the searcher is responsible for attaching a target
    bucket to an already content-anchored query.
    """
    query = str(raw_query or "").strip().lower()
    tokens = _content_query_tokens(query)

    if tokens and all(token in _DIFFICULTY_QUERY_TOKENS for token in tokens):
        fallback_tokens = _content_query_tokens(fallback)
        fallback_has_content = any(
            token not in _DIFFICULTY_QUERY_TOKENS
            for token in fallback_tokens
        )
        if fallback_has_content:
            return normalize_content_search_query(fallback, goal=goal, fallback="")
        return fallback_query_from_goal(goal)

    if not tokens:
        base = fallback or fallback_query_from_goal(goal)
        if base and base.lower() != query:
            return normalize_content_search_query(base, goal=goal, fallback="")
        return fallback_query_from_goal(goal)

    return " ".join(tokens[:4])
