"""Tests for the LLM-cost reduction toggles in :mod:`nare.config.CostConfig`.

Each test isolates one toggle and verifies that the documented LLM-call
saving actually happens (or does not happen, when the toggle is off).
None of these tests hit a real LLM endpoint — they patch the relevant
network entry points.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import patch

from nare import compat as llm
from nare.compat import HybridCritic
from nare.config import DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# P1.1 — Embedding LRU cache
# ---------------------------------------------------------------------------


def test_embedding_cache_dedupes_identical_queries():
    llm.clear_embedding_cache()

    calls = []

    def fake_post(url, payload, retries=5):
        calls.append(payload["content"]["parts"][0]["text"])
        return {"embedding": {"values": [0.1, 0.2, 0.3]}}

    with patch.object(llm, "_post", side_effect=fake_post), patch.object(
        llm, "API_KEY", "fake-key"
    ):
        v1 = llm.get_embedding("hello")
        v2 = llm.get_embedding("hello")
        v3 = llm.get_embedding("world")

    assert v1 == v2
    assert len(calls) == 2  # two distinct queries -> two API calls
    assert "hello" in calls
    assert "world" in calls


def test_embedding_cache_disabled_when_size_zero():
    llm.clear_embedding_cache()
    cfg = dataclasses.replace(
        DEFAULT_CONFIG,
        cost=dataclasses.replace(DEFAULT_CONFIG.cost, embedding_cache_size=0),
    )
    calls = []

    def fake_post(url, payload, retries=5):
        calls.append(1)
        return {"embedding": {"values": [0.0]}}

    with patch.object(llm, "DEFAULT_CONFIG", cfg), patch.object(
        llm, "_post", side_effect=fake_post
    ), patch.object(llm, "API_KEY", "fake-key"):
        llm.get_embedding("hello")
        llm.get_embedding("hello")

    # Cache disabled -> two separate calls
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# P1.2 — Skip pairwise Elo when every candidate has gt_score
# ---------------------------------------------------------------------------


def _code_cands():
    """All three candidates contain compilable Python (so gt_score is set)."""
    return [
        {"solution": "def f():\n    return 1\n", "reasoning": "", "tot_score": 5.0},
        {"solution": "def g():\n    return 2\n", "reasoning": "", "tot_score": 4.0},
        {"solution": "def h():\n    print(99)\n    return 3\n", "reasoning": "", "tot_score": 3.0},
    ]


def test_critic_skips_pairwise_when_all_gt_scores_set():
    critic = HybridCritic(DEFAULT_CONFIG)
    calls = []

    def fake_judge(*args, **kwargs):
        calls.append(1)
        return 1

    with patch("nare.agent.llm.llm_pairwise_judge", side_effect=fake_judge):
        result = critic.evaluate("test", _code_cands())

    assert all(c.get("gt_score") is not None for c in result)
    assert len(calls) == 0


def test_critic_runs_pairwise_when_toggle_disabled():
    cfg = dataclasses.replace(
        DEFAULT_CONFIG,
        cost=dataclasses.replace(
            DEFAULT_CONFIG.cost, skip_pairwise_when_gt_available=False
        ),
    )
    critic = HybridCritic(cfg)
    calls = []

    def fake_judge(*args, **kwargs):
        calls.append(1)
        return 1

    with patch("nare.agent.llm.llm_pairwise_judge", side_effect=fake_judge):
        critic.evaluate("test", _code_cands())

    # n=3 candidates -> 2 pairwise calls.
    assert len(calls) == 2


def test_critic_runs_pairwise_when_some_candidates_lack_gt_score():
    """Mixed code + prose -> some gt_score is None, must fall back to LLM."""
    critic = HybridCritic(DEFAULT_CONFIG)
    cands = [
        {"solution": "The answer is 7.", "reasoning": "", "tot_score": 5.0},
        {"solution": "def f():\n    return 7\n", "reasoning": "", "tot_score": 4.0},
    ]

    calls = []

    def fake_judge(*args, **kwargs):
        calls.append(1)
        return 1

    with patch("nare.agent.llm.llm_pairwise_judge", side_effect=fake_judge):
        critic.evaluate("test", cands)

    # n=2 -> 1 pairwise call
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# P1.4 — Local heuristic ToT ranker
# ---------------------------------------------------------------------------


def test_local_thought_score_ranks_code_above_failure():
    rich = "def f(x):\n    return x + 1\n# implements the formula"
    failure = "I cannot solve this. Impossible. No idea."
    assert llm._local_thought_score(rich) > llm._local_thought_score(failure)


def test_local_thought_score_bounded_0_10():
    assert 0.0 <= llm._local_thought_score("") <= 10.0
    huge = "def f():\n    return 1\n" * 1000
    assert 0.0 <= llm._local_thought_score(huge) <= 10.0
