"""Regression test for HYBRID oracle-veto on save.

Before this PR, the HYBRID branch in :meth:`NAREProductionAgent.solve`
called ``_save_episode`` unconditionally — even if the per-task oracle
had adjudicated the candidate as wrong. This let an oracle-vetoed
HYBRID answer (e.g. ``noitgnic`` for a "reverse cognition" task) seed
the FAST cache and corrupt downstream paraphrase hits.

The SLOW path was already protected by the
``final_score >= max(0.70, mem_avg)`` threshold, which is unreachable
when ``gt_score=0.0`` because the weighted final score caps at
``0.25 + 0.20 + 0.15 = 0.60`` without oracle approval. HYBRID was
missing this guard.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from nare.core.agent import NAREProductionAgent
from nare import llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_hybrid_band_agent():
    """Return an agent configured so that one mocked retrieval lands
    in the HYBRID similarity band (between tau_hybrid and tau_fast)."""
    agent = NAREProductionAgent()
    # Force a wide HYBRID band so our planted similarity (0.85) sits
    # cleanly inside it regardless of config defaults.
    agent.tau_fast = 0.99
    agent.tau_hybrid = 0.30
    agent.memory.episodes = []
    agent.memory.semantic_rules = []
    return agent


class _FakeIndex:
    """Stand-in for FAISS HNSW: returns a low similarity so the
    Layer-0 FAST early return is skipped, and the routing decision
    falls through to the explicit HYBRID branch."""

    def __init__(self, dim: int):
        self.d = dim
        self.ntotal = 1  # non-zero so the early-FAST block executes

    def search(self, vec, k):
        # Return sim well below tau_fast so the early-FAST gate misses.
        sims = np.array([[0.10] * k], dtype=np.float32)
        idx = np.array([[0] * k], dtype=np.int64)
        return sims, idx


def _patch_for_hybrid(agent, dim: int, candidate_solution: str):
    """Set up the agent so a single solve() call reliably reaches the
    HYBRID save block with the supplied candidate solution."""
    fake_ep = {
        "query": "Reverse the string 'cognition'",
        "solution": "noitingoc",
        "embedding": np.array([0.5] * dim, dtype=np.float32),
        "reasoning_trace": "Built-in reverse",
        "id": 0,
        "score": 0.8,
        "similarity": 0.85,
    }
    agent.memory.episodic_index = _FakeIndex(dim)

    # Routing path retrieve_episodes — returns our HYBRID-band ep.
    def _stub_retrieve_episodes(vec, k=3):
        return [dict(fake_ep)]

    agent.memory.retrieve_episodes = _stub_retrieve_episodes  # type: ignore[assignment]
    # Other retrieval helpers must return empties so we don't hit them.
    agent.memory.retrieve_semantics = lambda vec, k=2: []  # type: ignore[assignment]
    agent.memory.retrieve_facts = lambda vec, k=3: []  # type: ignore[assignment]

    # rl_retriever.rerank is a no-op pass-through here.
    agent.rl_retriever.rerank = lambda eps: eps  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


DIM = 3072  # matches RLRetriever.embedding_dim default


def test_hybrid_save_blocked_when_oracle_vetoes_candidate():
    agent = _build_hybrid_band_agent()
    dim = DIM
    _patch_for_hybrid(agent, dim, candidate_solution="noitgnic")

    with patch.object(llm, "generate_samples", return_value=([{"solution": "noitgnic"}], 50)), \
         patch.object(llm, "get_embedding", return_value=[0.5] * dim), \
         patch.object(llm, "extract_heuristic_rule", return_value=None), \
         patch.object(agent, "_save_episode") as mock_save:
        oracle_spec = {
            "type": "string_contains",
            "must_contain": ["noitingoc"],
        }
        res = agent.solve("Reverse 'cognition'", oracle_spec=oracle_spec)

    assert res["route_decision"] == "HYBRID", (
        f"Expected HYBRID route, got {res['route_decision']!r}. The test "
        f"setup didn't land sim in the HYBRID band."
    )
    assert mock_save.call_count == 0, (
        "Oracle vetoed the HYBRID candidate (gt_score==0.0) but "
        "_save_episode was still called. This is the bug PR #17 fixes."
    )
    log_text = "\n".join(res.get("memory_update_log", []))
    assert "rejected by oracle" in log_text.lower(), (
        f"Expected 'rejected by oracle' note in log, got:\n{log_text}"
    )


def test_hybrid_save_proceeds_when_oracle_approves_candidate():
    agent = _build_hybrid_band_agent()
    dim = DIM
    _patch_for_hybrid(agent, dim, candidate_solution="noitingoc")

    with patch.object(llm, "generate_samples", return_value=([{"solution": "noitingoc"}], 50)), \
         patch.object(llm, "get_embedding", return_value=[0.5] * dim), \
         patch.object(llm, "extract_heuristic_rule", return_value=None), \
         patch.object(agent, "_save_episode", return_value=True) as mock_save:
        oracle_spec = {
            "type": "string_contains",
            "must_contain": ["noitingoc"],
        }
        res = agent.solve("Reverse 'cognition'", oracle_spec=oracle_spec)

    assert res["route_decision"] == "HYBRID"
    assert mock_save.call_count == 1, (
        "HYBRID with oracle-approved candidate should still save the "
        "episode (only oracle-rejected candidates are blocked)."
    )


def test_hybrid_save_proceeds_when_no_oracle_supplied():
    agent = _build_hybrid_band_agent()
    dim = DIM
    _patch_for_hybrid(agent, dim, candidate_solution="anything")

    with patch.object(llm, "generate_samples", return_value=([{"solution": "anything"}], 50)), \
         patch.object(llm, "get_embedding", return_value=[0.5] * dim), \
         patch.object(llm, "extract_heuristic_rule", return_value=None), \
         patch.object(agent, "_save_episode", return_value=True) as mock_save:
        res = agent.solve("Reverse 'cognition'")  # no oracle_spec

    assert res["route_decision"] == "HYBRID"
    assert mock_save.call_count == 1, (
        "Without oracle_spec there is no gt_score, so the legacy "
        "unconditional save behaviour must be preserved."
    )
