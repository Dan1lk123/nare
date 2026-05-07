"""Regression tests for the sleep-phase crash surfaced by the
real_100_benchmark log:

    [Pruning] low-quality data collected 1 dead rules: ['Reverse String (Enhanced)']
    [Sleep Phase] Failed: list assignment index out of range

The failure was that ``HybridMemory.update_semantic_rule(idx, ...)``
indexed ``self.semantic_rules[idx]`` unconditionally.  When two sleep
cycles ran concurrently (the agent kicks off sleep on a daemon thread),
or when ``_prune_weak_rules`` removed a rule between the caller's
``retrieve_semantics`` and the eventual ``update_semantic_rule``, the
positional index became stale and the whole sleep phase aborted via
the outer ``except Exception``.
"""

import os
import tempfile

import numpy as np
import pytest

from nare.config import DEFAULT_CONFIG
from nare.memory import MemorySystem


@pytest.fixture
def memory():
    """Fresh on-disk memory backed by a tmp directory."""
    with tempfile.TemporaryDirectory() as tmp:
        mem = MemorySystem(
            embedding_dim=64,
            persist_dir=tmp,
            config=DEFAULT_CONFIG,
        )
        yield mem


def _make_rule(pattern: str) -> dict:
    return {
        "pattern": pattern,
        "python_code": "def trigger(q): return False\n\ndef execute(q): return q",
        "confidence": 0.5,
        "success_count": 0,
    }


def _emb(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v


def test_update_returns_false_on_out_of_range(memory):
    """Calling update on an index past the end must NOT raise."""
    memory.add_semantic_rule(_make_rule("A"), _emb(0, memory.embedding_dim))
    assert len(memory.semantic_rules) == 1

    ok = memory.update_semantic_rule(
        idx=99,
        rule_data=_make_rule("ghost"),
        new_embedding=_emb(1, memory.embedding_dim),
    )
    assert ok is False
    # Original rule untouched.
    assert memory.semantic_rules[0]["pattern"] == "A"


def test_update_returns_false_when_pattern_mismatched(memory):
    """If the rule at idx no longer matches expected_pattern (e.g. the
    pruner removed it and another rule shifted into that slot), reject."""
    memory.add_semantic_rule(_make_rule("Reverse String"), _emb(0, memory.embedding_dim))
    memory.add_semantic_rule(_make_rule("Sum Even Numbers"), _emb(1, memory.embedding_dim))

    # Pretend the pruner removed 'Reverse String' so 'Sum Even Numbers'
    # is now at idx 0 — caller's stale index would have pointed at 0
    # expecting 'Reverse String'.
    memory.semantic_rules.pop(0)

    ok = memory.update_semantic_rule(
        idx=0,
        rule_data=_make_rule("Reverse String (Enhanced)"),
        new_embedding=_emb(2, memory.embedding_dim),
        expected_pattern="Reverse String",
    )
    assert ok is False
    # The remaining rule must not have been corrupted.
    assert memory.semantic_rules[0]["pattern"] == "Sum Even Numbers"


def test_update_succeeds_with_matching_pattern(memory):
    """Sanity: a valid update applies and returns True."""
    memory.add_semantic_rule(_make_rule("A"), _emb(0, memory.embedding_dim))

    new = _make_rule("A")
    new["confidence"] = 0.9
    new["pattern"] = "A"

    ok = memory.update_semantic_rule(
        idx=0,
        rule_data=new,
        new_embedding=_emb(3, memory.embedding_dim),
        expected_pattern="A",
    )
    assert ok is True
    assert memory.semantic_rules[0]["confidence"] == pytest.approx(0.9)
