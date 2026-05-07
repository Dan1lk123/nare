"""Honest naming for the SLOW-path search strategy.

The historical ``llm.tree_of_thoughts`` is not a real Tree-of-Thoughts
search — with ``depth=1`` (the default used by the agent), it is
exactly best-of-N candidate generation with LLM pre-scoring. We expose
that under :func:`nare.llm.best_of_n_with_prescore` and confirm:

1. The honest alias exists and is callable.
2. It dispatches to the same backend as ``tree_of_thoughts(..., depth=1)``.
3. The historical name still resolves (no API break).
"""

from __future__ import annotations

import inspect

from nare import compat as llm


def test_best_of_n_alias_is_present():
    assert hasattr(llm, "best_of_n_with_prescore")
    assert callable(llm.best_of_n_with_prescore)


def test_tree_of_thoughts_still_present():
    """We deliberately keep this for backward compatibility."""
    assert hasattr(llm, "tree_of_thoughts")
    assert callable(llm.tree_of_thoughts)


def test_best_of_n_signature_is_simpler():
    """The honest alias drops the misleading ``depth`` parameter."""
    sig = inspect.signature(llm.best_of_n_with_prescore)
    params = sig.parameters
    assert "depth" not in params
    assert "breadth" in params
    assert "prompt" in params


def test_best_of_n_dispatches_to_tree_of_thoughts(monkeypatch):
    """Calling ``best_of_n_with_prescore(prompt, breadth)`` should
    forward to ``tree_of_thoughts(prompt, breadth, depth=1)`` exactly."""
    captured = {}

    def fake_tree(prompt, breadth=3, depth=2):
        captured["args"] = (prompt, breadth, depth)
        return ([], 0)

    monkeypatch.setattr(llm, "tree_of_thoughts", fake_tree)
    llm.best_of_n_with_prescore("hello world", breadth=4)
    assert captured["args"] == ("hello world", 4, 1)
