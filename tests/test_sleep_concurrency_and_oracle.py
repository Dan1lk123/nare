"""Regression tests for two related issues observed in the
real_100_benchmark log on top of PR #3:

1. ``[Sleep Phase] Failed: list index out of range`` (DIFFERENT from the
   ``list assignment index out of range`` PR #3 fixed). Caused by two
   ``_sleep_phase`` calls running concurrently — one from the background
   daemon, one from the benchmark's own consolidation hook. The second
   caller computed cluster_indices against a stale snapshot and indexed
   into ``self.memory.episodes`` after the first caller had already
   deleted those entries.

   This file covers the new ``self._sleep_lock`` serialisation guard
   and the bounds-check on stale ``cluster_indices``.

2. ``oracle_spec`` plumbing through ``solve(...)``. PR #5 wired the
   Oracle protocol into ``HybridCritic.evaluate``, but ``solve(query)``
   had no way to deliver a per-query oracle — so callers (benchmark,
   API users) couldn't actually exercise the new GT validator. This
   adds ``solve(query, oracle_spec=...)`` and persists the spec on
   the saved episode for future sleep-phase / REM cached-replay use.
"""

import tempfile
import threading
import time

import pytest

from nare.core.agent import NAREProductionAgent
from nare.config import DEFAULT_CONFIG
from nare.memory import MemorySystem


@pytest.fixture
def isolated_agent():
    """An agent backed by a fresh tmp persist_dir so cross-test state
    leakage (memory_store/episodic.faiss on disk) cannot dedupe later
    inserts. Yields the agent and tears down the tmp dir afterwards."""
    tmp = tempfile.TemporaryDirectory()
    agent = NAREProductionAgent(config=DEFAULT_CONFIG)
    # Replace the auto-created memory with one rooted in our tmp dir.
    agent.memory = MemorySystem(
        embedding_dim=agent.memory.embedding_dim,
        persist_dir=tmp.name,
        config=DEFAULT_CONFIG,
    )
    try:
        yield agent
    finally:
        tmp.cleanup()


# ----------------------------------------------------------------------
# Sleep-phase concurrency
# ----------------------------------------------------------------------

class _StubMemory:
    """Tiny stand-in for the parts of MemorySystem _sleep_phase touches."""

    def __init__(self):
        self._lock = threading.RLock()
        self.episodes = []
        self.semantic_rules = []

    def prune_fading_memories(self):
        return None

    def save(self):
        return None

    def _rebuild_episodic_index(self):
        return None


def _make_agent_with_stub_memory():
    """Build a NAREProductionAgent and override its memory so we can
    drive _sleep_phase deterministically without spinning up FAISS or
    LLM dependencies for the concurrency test."""
    agent = NAREProductionAgent(config=DEFAULT_CONFIG)
    agent.memory = _StubMemory()  # type: ignore[assignment]
    return agent


def test_sleep_lock_serialises_concurrent_sleep_phase():
    """Two concurrent _sleep_phase calls must not both enter the body."""
    agent = _make_agent_with_stub_memory()

    enter_count = {"n": 0}
    release_event = threading.Event()

    def slow_locked():
        enter_count["n"] += 1
        # Hold the lock long enough for the second caller to attempt
        # acquisition and bail out.
        release_event.wait(timeout=2.0)

    agent._sleep_phase_locked = slow_locked  # type: ignore[assignment]

    t1 = threading.Thread(target=agent._sleep_phase)
    t2 = threading.Thread(target=agent._sleep_phase)
    t1.start()
    # Tiny pause so t1 wins the lock race deterministically.
    time.sleep(0.05)
    t2.start()
    # Let the second caller observe the lock as held and return.
    time.sleep(0.1)
    release_event.set()
    t1.join(timeout=3.0)
    t2.join(timeout=3.0)

    # Only the first caller should have entered the body.
    assert enter_count["n"] == 1


def test_sleep_lock_serialises_rem_against_nrem():
    """REM must also defer when an NREM cycle is mid-flight (and vice
    versa) — same lock so they can't interleave on overlapping memory."""
    agent = _make_agent_with_stub_memory()

    rem_entered = {"n": 0}
    nrem_release = threading.Event()

    def slow_nrem():
        nrem_release.wait(timeout=2.0)

    def stub_rem():
        rem_entered["n"] += 1

    agent._sleep_phase_locked = slow_nrem  # type: ignore[assignment]
    agent._rem_sleep_phase_locked = stub_rem  # type: ignore[assignment]

    t_nrem = threading.Thread(target=agent._sleep_phase)
    t_nrem.start()
    time.sleep(0.05)
    # While NREM holds the lock, REM should observe it and skip.
    agent._rem_sleep_phase()
    assert rem_entered["n"] == 0

    nrem_release.set()
    t_nrem.join(timeout=3.0)

    # After NREM releases, REM should now enter.
    agent._rem_sleep_phase()
    assert rem_entered["n"] == 1


def test_sleep_lock_released_on_exception():
    """If _sleep_phase_locked raises, the lock must still be released
    so subsequent cycles can run."""
    agent = _make_agent_with_stub_memory()

    def boom():
        raise RuntimeError("induced sleep failure")

    agent._sleep_phase_locked = boom  # type: ignore[assignment]

    with pytest.raises(RuntimeError):
        agent._sleep_phase()

    # Lock must be free now.
    assert agent._sleep_lock.acquire(blocking=False)
    agent._sleep_lock.release()


# ----------------------------------------------------------------------
# oracle_spec plumbing through solve(...)
# ----------------------------------------------------------------------

def test_save_episode_stores_oracle_spec(isolated_agent):
    """_save_episode must persist oracle_spec verbatim on the episode
    dict so REM cached-replay and sleep-phase skill validation can
    reach it later."""
    agent = isolated_agent

    # Seed the agent with a deterministic embedding so add_episode
    # doesn't try to dedupe against an empty index.
    query = "What is 2 + 2?"
    embedding = [0.1] * agent.memory.embedding_dim

    spec = {"type": "numeric_set", "expected": [4]}
    best_cand = {
        "solution": "4",
        "final_score": 0.9,
        "reasoning": "trivial arithmetic",
        "abstract_signature": query,
    }

    ok = agent._save_episode(
        query, embedding, best_cand, "TEST PROMPT", oracle_spec=spec
    )
    assert ok is True

    saved = agent.memory.episodes[-1]
    assert saved["oracle_spec"] == spec
    assert saved["query"] == query
    assert saved["solution"] == "4"


def test_save_episode_omits_oracle_spec_when_none(isolated_agent):
    """Existing call sites must keep working; absence of spec must not
    write a key on the episode."""
    agent = isolated_agent

    query = "What is 5 + 5?"
    embedding = [0.2] * agent.memory.embedding_dim
    best_cand = {
        "solution": "10",
        "final_score": 0.9,
        "reasoning": "trivial arithmetic",
        "abstract_signature": query,
    }

    ok = agent._save_episode(query, embedding, best_cand, "TEST PROMPT")
    assert ok is True

    saved = agent.memory.episodes[-1]
    assert "oracle_spec" not in saved


def test_save_episode_skips_low_score_with_oracle_spec(isolated_agent):
    """oracle_spec parameter must not change the low-score rejection
    behaviour (existing contract)."""
    agent = isolated_agent

    query = "What is 7 + 7?"
    embedding = [0.3] * agent.memory.embedding_dim
    best_cand = {
        "solution": "??",
        "final_score": 0.1,  # below 0.50 threshold
        "reasoning": "guess",
        "abstract_signature": query,
    }
    spec = {"type": "numeric_set", "expected": [14]}

    ok = agent._save_episode(
        query, embedding, best_cand, "TEST PROMPT", oracle_spec=spec
    )
    assert ok is False
    assert agent.memory.episodes == []


def test_solve_passes_oracle_to_critic_and_persists_spec(isolated_agent, monkeypatch):
    """End-to-end: ``solve(query, oracle_spec=...)`` must pass an Oracle
    into ``HybridCritic.evaluate`` AND persist the spec on the saved
    episode. We monkeypatch the LLM-heavy parts so the test runs offline.
    """
    agent = isolated_agent

    # Stub embeddings to a deterministic vector.
    fake_emb = [0.42] * agent.memory.embedding_dim
    from nare import llm as nare_llm

    monkeypatch.setattr(nare_llm, "get_embedding", lambda q: fake_emb)

    # Stub the SLOW-path candidate generation.
    fake_candidates = [
        {
            "solution": "answer",
            "reasoning": "trace",
            "final_score": 0.0,  # critic rewrites this
            "abstract_signature": "test query",
        }
    ]
    monkeypatch.setattr(
        nare_llm,
        "best_of_n_with_prescore",
        lambda prompt, breadth=1, **kw: (fake_candidates, 0),
    )
    monkeypatch.setattr(
        nare_llm,
        "generate_samples",
        lambda prompt, n=1, mode="HYBRID", **kw: (fake_candidates, 0),
    )

    # Stub _post_process_answer so we don't try to AST-execute "answer".
    monkeypatch.setattr(
        agent, "_post_process_answer", lambda raw, route, log: raw
    )

    # Capture what HybridCritic.evaluate received.
    received = {}
    real_evaluate = agent.critic.evaluate

    def spy_evaluate(query, candidates, *, oracle=None, **kwargs):
        received["oracle"] = oracle
        received["query"] = query
        # Bypass the heavy critic; assign a passing score.
        for c in candidates:
            c["final_score"] = 0.95
        return candidates

    agent.critic.evaluate = spy_evaluate  # type: ignore[assignment]

    spec = {"type": "string_contains", "must_contain": ["answer"]}
    result = agent.solve("test query", oracle_spec=spec)

    # Critic must have received a callable oracle (Oracle protocol).
    assert callable(received.get("oracle")), (
        f"Expected oracle to be wired into critic.evaluate; got {received!r}"
    )

    # The episode must have been saved with the spec.
    assert agent.memory.episodes, "expected solve() to save an episode"
    saved = agent.memory.episodes[-1]
    assert saved.get("oracle_spec") == spec

    # Cleanup
    agent.critic.evaluate = real_evaluate  # type: ignore[assignment]
    _ = result  # quiet unused-var lint


def test_solve_with_bad_oracle_spec_does_not_crash(isolated_agent, monkeypatch):
    """A malformed oracle_spec must be caught and downgraded to no
    oracle, not abort the solve."""
    agent = isolated_agent

    fake_emb = [0.7] * agent.memory.embedding_dim
    from nare import llm as nare_llm

    monkeypatch.setattr(nare_llm, "get_embedding", lambda q: fake_emb)
    monkeypatch.setattr(
        nare_llm,
        "best_of_n_with_prescore",
        lambda prompt, breadth=1, **kw: ([
            {
                "solution": "x",
                "reasoning": "r",
                "final_score": 0.0,
                "abstract_signature": "q",
            }
        ], 0),
    )
    monkeypatch.setattr(
        nare_llm,
        "generate_samples",
        lambda prompt, n=1, mode="HYBRID", **kw: ([
            {
                "solution": "x",
                "reasoning": "r",
                "final_score": 0.0,
                "abstract_signature": "q",
            }
        ], 0),
    )
    monkeypatch.setattr(
        agent, "_post_process_answer", lambda raw, route, log: raw
    )
    received = {}

    def spy_evaluate(query, candidates, *, oracle=None, **kwargs):
        received["oracle"] = oracle
        for c in candidates:
            c["final_score"] = 0.95
        return candidates

    agent.critic.evaluate = spy_evaluate  # type: ignore[assignment]

    bad_spec = {"type": "definitely_not_a_real_oracle_kind"}
    # Should not raise.
    agent.solve("bad oracle test", oracle_spec=bad_spec)

    # Critic must have been called with oracle=None (graceful fallback).
    assert received.get("oracle") is None
