"""Integration tests for ``NAREProductionAgent.calibrate_tau_fast_from_replay``.

These tests stub MemorySystem so we don't need Gemini. We bind the
calibration method directly off the real class to exercise its
locking, snapshot, and threshold-clamping logic.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from nare.core.agent import NAREProductionAgent
from nare.config import DEFAULT_CONFIG


class _FakeMemory:
    def __init__(self, episodes):
        self.episodes = list(episodes)
        self._lock = threading.RLock()


class _FakeAgent:
    def __init__(self, episodes, *, tau_fast=0.98, tau_min=0.95, tau_max=0.99,
                 config=DEFAULT_CONFIG):
        self.memory = _FakeMemory(episodes)
        self.config = config
        self.tau_fast = tau_fast
        self.tau_min = tau_min
        self.tau_max = tau_max

    calibrate_tau_fast_from_replay = NAREProductionAgent.calibrate_tau_fast_from_replay


def _ep(query: str, solution: str, vec):
    arr = np.array(vec, dtype=np.float32)
    arr = arr / (np.linalg.norm(arr) + 1e-12)
    return {"query": query, "solution": solution, "embedding": arr.tolist()}


def test_no_episodes_no_recommendation():
    agent = _FakeAgent([])
    report = agent.calibrate_tau_fast_from_replay(target_precision=0.95)
    assert report["recommended_tau"] is None
    assert agent.tau_fast == pytest.approx(0.98)


def test_perfect_cluster_lowers_tau_to_clamp():
    """All embeddings identical, all solutions identical: every tau is
    perfectly precise. The recommendation will be the lowest in the
    sweep, but the agent clamps to its tau_min config band."""
    eps = [_ep(f"q{i}", "42", [1.0, 0.0]) for i in range(4)]
    agent = _FakeAgent(eps, tau_fast=0.98, tau_min=0.95, tau_max=0.99)
    report = agent.calibrate_tau_fast_from_replay(
        target_precision=0.95, tau_min=0.50, tau_step=0.01
    )
    assert report["recommended_tau"] is not None
    # Unclamped recommendation can be as low as 0.50; clamped tau must
    # not drop below the agent's tau_min band.
    assert agent.tau_fast >= 0.95
    assert agent.tau_fast <= 0.99


def test_apply_false_does_not_mutate():
    eps = [_ep(f"q{i}", "42", [1.0, 0.0]) for i in range(3)]
    agent = _FakeAgent(eps, tau_fast=0.98)
    original = agent.tau_fast
    report = agent.calibrate_tau_fast_from_replay(
        target_precision=0.95, apply=False, tau_min=0.50, tau_step=0.01
    )
    assert report["recommended_tau"] is not None
    assert agent.tau_fast == pytest.approx(original)


def test_low_precision_population_does_not_mutate():
    """Two near-identical vectors with different solutions: no tau
    achieves target precision; the agent does not lower tau_fast."""
    eps = [
        _ep("q1", "42", [1.0, 0.001]),
        _ep("q2", "99", [1.0, -0.001]),
    ]
    agent = _FakeAgent(eps, tau_fast=0.98)
    original = agent.tau_fast
    report = agent.calibrate_tau_fast_from_replay(
        target_precision=0.95, tau_min=0.50, tau_step=0.01
    )
    assert report["recommended_tau"] is None
    assert agent.tau_fast == pytest.approx(original)
