"""Tests for HybridCritic._ground_truth_validate.

The legacy validator returned 0.8 for any syntactically valid Python and
0.2 otherwise — so ``def f(): return 99999`` and a real correct
implementation got the *same* score. This test suite verifies the two
new modes:

* **Executability mode** — runs the candidate via the AST sandbox and
  scores by whether ``execute(query)`` produces a non-error string.

* **Oracle mode** — when an Oracle is supplied, the binary verdict from
  the oracle is the ``gt_score``.

The legacy AST-only path is still reachable via
``CriticConfig(gt_use_execution=False)`` for benchmark rollback.
"""

from __future__ import annotations

import dataclasses

from nare.compat import HybridCritic
from nare.config import DEFAULT_CONFIG, NareConfig


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


_VALID_SKILL = """\
def trigger(query):
    return True

def execute(query):
    return "hello"
"""

_VALID_SKILL_THAT_RAISES = """\
def trigger(query):
    return True

def execute(query):
    raise ValueError("boom")
"""

_VALID_PYTHON_NOT_A_SKILL = """\
def f():
    return 99999
"""

# `_extract_code` requires both ``def `` AND (``return`` or ``print``)
# to consider a candidate "code"; this string satisfies both yet has a
# real syntax error inside, so it reaches the AST/load path.
_SYNTAX_ERROR = "def f(:\n    return 1\n"


def _critic_with_execution(enabled: bool) -> HybridCritic:
    cfg = dataclasses.replace(
        DEFAULT_CONFIG,
        critic=dataclasses.replace(DEFAULT_CONFIG.critic, gt_use_execution=enabled),
    )
    return HybridCritic(cfg)


# ---------------------------------------------------------------------
# Default executability mode
# ---------------------------------------------------------------------


def test_executable_skill_scores_high():
    critic = _critic_with_execution(True)
    score = critic._ground_truth_validate(
        _VALID_SKILL, query="anything",
    )
    assert score == 0.85, (
        f"Expected 0.85 for a runnable skill, got {score}"
    )


def test_skill_that_raises_at_execute_scores_lower():
    """A NARE skill whose ``execute`` raises must score below the
    success ceiling — this is the central property the legacy AST path
    couldn't express."""
    critic = _critic_with_execution(True)
    score = critic._ground_truth_validate(
        _VALID_SKILL_THAT_RAISES, query="x",
    )
    # safe_call_execute_in_namespace catches the exception and returns
    # an "Error: ..." string, so executability mode caps at 0.40.
    assert 0.0 <= score <= 0.55, (
        f"Skill that raises should not score 0.85, got {score}"
    )
    assert score < 0.85


def test_valid_python_without_skill_api_scores_partial():
    critic = _critic_with_execution(True)
    score = critic._ground_truth_validate(
        _VALID_PYTHON_NOT_A_SKILL, query="x",
    )
    # 0.55: it loads, but doesn't expose execute(); honest middle
    # ground.
    assert score == 0.55


def test_syntax_error_scores_low():
    critic = _critic_with_execution(True)
    score = critic._ground_truth_validate(_SYNTAX_ERROR, query="x")
    assert score == 0.20


def test_non_code_returns_none():
    """When there's no detectable code at all and no oracle is given,
    the validator returns None so the LLM critic can take over."""
    critic = _critic_with_execution(True)
    score = critic._ground_truth_validate(
        "The answer is 42.", query="what is 6 * 7?",
    )
    assert score is None


# ---------------------------------------------------------------------
# Legacy AST-only fallback (gt_use_execution=False)
# ---------------------------------------------------------------------


def test_legacy_path_returns_08_for_valid_syntax():
    """``CriticConfig(gt_use_execution=False)`` reproduces the old
    behaviour: any AST-valid Python scores 0.8 regardless of what it
    does. This is intentional for benchmark rollback."""
    critic = _critic_with_execution(False)
    score = critic._ground_truth_validate(
        _VALID_PYTHON_NOT_A_SKILL, query="x",
    )
    assert score == 0.8


def test_legacy_path_returns_02_for_invalid_syntax():
    critic = _critic_with_execution(False)
    score = critic._ground_truth_validate(_SYNTAX_ERROR, query="x")
    assert score == 0.2


# ---------------------------------------------------------------------
# Oracle mode
# ---------------------------------------------------------------------


def test_oracle_mode_returns_10_on_accept():
    critic = _critic_with_execution(True)
    accept_oracle = lambda q, a: (True, "ok")
    score = critic._ground_truth_validate(
        "anything", query="q", oracle=accept_oracle,
    )
    assert score == 1.0


def test_oracle_mode_returns_00_on_reject():
    critic = _critic_with_execution(True)
    reject_oracle = lambda q, a: (False, "wrong")
    score = critic._ground_truth_validate(
        _VALID_SKILL, query="q", oracle=reject_oracle,
    )
    assert score == 0.0


def test_oracle_overrides_executability():
    """Even a runnable skill must score 0.0 if the oracle rejects it.
    This is the entire point of pluggable oracles — the GT signal must
    reflect the oracle, not 'did the code run'."""
    critic = _critic_with_execution(True)
    reject_oracle = lambda q, a: (False, "wrong answer")
    score = critic._ground_truth_validate(
        _VALID_SKILL, query="q", oracle=reject_oracle,
    )
    assert score == 0.0


def test_oracle_crash_falls_back_to_executability():
    critic = _critic_with_execution(True)

    def crashing_oracle(q, a):
        raise RuntimeError("oracle blew up")

    score = critic._ground_truth_validate(
        _VALID_SKILL, query="q", oracle=crashing_oracle,
    )
    # Must NOT propagate; falls through to executability mode.
    assert score == 0.85


# ---------------------------------------------------------------------
# evaluate() integration
# ---------------------------------------------------------------------


def test_evaluate_passes_oracle_through():
    critic = HybridCritic(DEFAULT_CONFIG)
    capture = {"calls": 0}

    def counting_oracle(q, a):
        capture["calls"] += 1
        return (True, "ok")

    candidates = [
        {"solution": _VALID_SKILL},
        {"solution": _VALID_SKILL_THAT_RAISES},
    ]
    # Stub out the LLM pairwise judge so the test stays offline.
    from nare import compat as llm_mod
    original = llm_mod.llm_pairwise_judge
    llm_mod.llm_pairwise_judge = lambda *a, **kw: 1
    try:
        critic.evaluate("query", candidates, oracle=counting_oracle)
    finally:
        llm_mod.llm_pairwise_judge = original

    # Oracle was consulted once per candidate.
    assert capture["calls"] == 2
    # Both gt_scores reflect the oracle verdict (1.0 for accept).
    assert candidates[0]["gt_score"] == 1.0
    assert candidates[1]["gt_score"] == 1.0


def test_evaluate_without_oracle_uses_executability():
    critic = HybridCritic(DEFAULT_CONFIG)
    candidates = [{"solution": _VALID_SKILL}]
    critic.evaluate("query", candidates)
    # Single-candidate path uses gt_score directly as final_score.
    assert candidates[0]["gt_score"] == 0.85
    assert candidates[0]["final_score"] == 0.85


# ---------------------------------------------------------------------
# Default config sanity
# ---------------------------------------------------------------------


def test_default_config_uses_execution():
    """Regression guard: the default must be the new (honest) path."""
    assert DEFAULT_CONFIG.critic.gt_use_execution is True
