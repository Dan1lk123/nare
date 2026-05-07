"""Regression tests for the ``HybridCritic.evaluate`` oracle parameter.

PR #8 added ``oracle=per_call_oracle`` to two call sites in
:func:`NAREProductionAgent.solve` (lines 1651 and 1732 in agent.py)
without updating the critic's signature, so every SLOW / HYBRID call
that materialised an oracle from ``oracle_spec`` would raise

    TypeError: HybridCritic.evaluate() got an unexpected keyword
    argument 'oracle'

This file pins:

1. The signature now accepts ``oracle`` (kwarg, default ``None``).
2. When ``oracle`` is supplied, the per-candidate ``gt_score`` is
   sourced from the oracle's verdict (1.0 / 0.0), not the AST/sandbox
   compilability heuristic.
3. When ``oracle`` is omitted, behaviour is unchanged — ``gt_score``
   keeps coming from the heuristic ``_ground_truth_validate``.
4. A faulty oracle that raises is caught and we fall back to the
   heuristic instead of crashing the whole solve.
"""

from __future__ import annotations

import inspect
from typing import Tuple

import pytest

from nare.compat import HybridCritic


# --- A tiny, deterministic oracle for these tests ------------------------


def _expecting(expected: str):
    def _oracle(query: str, candidate: str) -> Tuple[bool, str]:
        return (expected in candidate, f"looking for {expected!r}")
    return _oracle


def _broken_oracle(query: str, candidate: str) -> Tuple[bool, str]:
    raise RuntimeError("oracle hot wire melted")


# --- 1. Signature ---------------------------------------------------------


def test_evaluate_signature_accepts_oracle_kwarg():
    sig = inspect.signature(HybridCritic.evaluate)
    assert "oracle" in sig.parameters
    assert sig.parameters["oracle"].default is None


def test_evaluate_does_not_typeerror_with_oracle_kwarg():
    """Direct regression for the exact error
    ``HybridCritic.evaluate() got an unexpected keyword argument
    'oracle'`` that PR #8 introduced."""
    critic = HybridCritic()
    critic.evaluate("q?", [{"solution": "anything"}], oracle=None)


# --- 2. Oracle drives gt_score when supplied ------------------------------


def test_oracle_verdict_overrides_heuristic_gt_for_single_candidate():
    critic = HybridCritic()
    cands = critic.evaluate(
        "What is 2+2?",
        [{"solution": "The answer is 4."}],
        oracle=_expecting("4"),
    )
    assert cands[0]["gt_score"] == 1.0
    assert cands[0]["final_score"] == 1.0


def test_oracle_negative_verdict_pulls_gt_to_zero_for_single_candidate():
    critic = HybridCritic()
    cands = critic.evaluate(
        "What is 2+2?",
        [{"solution": "The answer is 5."}],
        oracle=_expecting("4"),
    )
    assert cands[0]["gt_score"] == 0.0


def test_oracle_separates_correct_from_incorrect_in_multi_candidate_path():
    """When the oracle is supplied the *correct* candidate must end up
    with a strictly higher final_score than the wrong one — that's the
    whole point of plumbing oracle_spec through PR #8."""
    critic = HybridCritic()
    cands = critic.evaluate(
        "What is 2+2?",
        [
            {"solution": "I think it might be 5."},
            {"solution": "The answer is 4."},
            {"solution": "Definitely 7."},
        ],
        oracle=_expecting("4"),
    )
    correct = next(c for c in cands if "4." in c["solution"])
    wrongs = [c for c in cands if c is not correct]
    assert correct["gt_score"] == 1.0
    assert all(w["gt_score"] == 0.0 for w in wrongs)
    assert correct["final_score"] > max(w["final_score"] for w in wrongs)


# --- 3. Backwards-compatible behaviour without oracle ---------------------


def test_evaluate_without_oracle_keeps_legacy_heuristic_gt_score():
    """A code-shaped solution must still get the AST-validated 0.8 from
    the heuristic when no oracle is supplied — this is the
    pre-PR-#8 contract that other callers (sleep / refinement) rely on.
    """
    critic = HybridCritic()
    code_solution = "```python\ndef f(x):\n    return x + 1\n```"
    cands = critic.evaluate("any?", [{"solution": code_solution}])
    # Heuristic: AST-valid code → gt_score ≈ 0.8
    assert cands[0]["gt_score"] == pytest.approx(0.8)


def test_evaluate_without_oracle_marks_non_code_solution_with_none_gt():
    critic = HybridCritic()
    cands = critic.evaluate("any?", [{"solution": "Just prose, no code here."}])
    assert cands[0]["gt_score"] is None


# --- 4. Faulty oracle handling --------------------------------------------


def test_broken_oracle_does_not_crash_solve_falls_back_to_heuristic():
    """If the oracle itself raises, the critic must NOT propagate the
    exception (we'd lose the candidate ranking AND crash solve). It
    falls back to the heuristic gt_score and logs a warning."""
    critic = HybridCritic()
    code_solution = "```python\ndef f(x): return x\n```"
    cands = critic.evaluate(
        "q?",
        [{"solution": code_solution}],
        oracle=_broken_oracle,
    )
    # Heuristic kicks in: AST-valid code → 0.8
    assert cands[0]["gt_score"] == pytest.approx(0.8)


# --- 5. Oracle ``info`` diagnostic preservation ---------------------------
#
# Oracles return ``(ok, info)`` where ``info`` is a free-form string
# (e.g. ``"missing expected numbers: [42]"``) intended for repair
# prompts and logging.  The original PR #14 fix discarded ``info``;
# this section pins the contract that callers can read it back via
# ``candidate['oracle_info']``.


def _expecting_with_info(expected: str, info_pass: str, info_fail: str):
    def _oracle(query: str, candidate: str) -> Tuple[bool, str]:
        ok = expected in candidate
        return (ok, info_pass if ok else info_fail)
    return _oracle


def test_oracle_info_preserved_on_single_candidate_pass():
    critic = HybridCritic()
    cands = critic.evaluate(
        "What is 2+2?",
        [{"solution": "The answer is 4."}],
        oracle=_expecting_with_info("4", "all expected numbers found", "missing 4"),
    )
    assert cands[0]["oracle_info"] == "all expected numbers found"


def test_oracle_info_preserved_on_single_candidate_fail():
    critic = HybridCritic()
    cands = critic.evaluate(
        "What is 2+2?",
        [{"solution": "The answer is 5."}],
        oracle=_expecting_with_info("4", "ok", "missing expected numbers: [4]"),
    )
    assert cands[0]["oracle_info"] == "missing expected numbers: [4]"


def test_oracle_info_per_candidate_in_multi_candidate_path():
    critic = HybridCritic()
    cands = critic.evaluate(
        "What is 2+2?",
        [
            {"solution": "I think it is 5."},
            {"solution": "The answer is 4."},
            {"solution": "Probably 7."},
        ],
        oracle=_expecting_with_info("4", "ok", "missing 4"),
    )
    correct = next(c for c in cands if "4." in c["solution"])
    wrongs = [c for c in cands if c is not correct]
    assert correct["oracle_info"] == "ok"
    assert all(w["oracle_info"] == "missing 4" for w in wrongs)


def test_no_oracle_means_no_oracle_info_key_added():
    """Backwards compatibility: callers that don't supply an oracle
    must not see ``oracle_info`` keys appearing on candidate dicts —
    that would surprise downstream code that checks for its presence."""
    critic = HybridCritic()
    cands = critic.evaluate(
        "q?",
        [{"solution": "anything"}, {"solution": "another"}],
    )
    for c in cands:
        assert "oracle_info" not in c


def test_broken_oracle_does_not_leave_stale_oracle_info_on_candidate():
    """If the oracle raised on this candidate the heuristic fallback is
    used, so no oracle_info should be attached."""
    critic = HybridCritic()
    cands = critic.evaluate(
        "q?",
        [{"solution": "```python\ndef f(): return 1\n```"}],
        oracle=_broken_oracle,
    )
    assert "oracle_info" not in cands[0]
