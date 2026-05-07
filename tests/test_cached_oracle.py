"""Tests for the strict cached-episode oracle (Tier A1).

The cached-episode oracle replaces ``heuristic_overlap`` as the default
fallback in ``_validate_skill``. Its contract:

* Normalized exact string match accepts.
* Numeric-set: every expected number must appear, AND no extra
  unrelated numbers are allowed.
* "Error:" prefix always rejects.
* Empty / "N/A" expected always rejects (we never promote a skill
  against an episode that has no verified ground truth).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from nare.config import DEFAULT_CONFIG, SkillValidationConfig
from nare.reasoning.generation.engine import _validate_skill
from nare.oracle import (
    build_oracle_from_spec,
    cached_episode_oracle,
)


# ---------------------------------------------------------------------------
# Direct oracle-level tests
# ---------------------------------------------------------------------------


def test_cached_oracle_exact_match_accepts():
    oracle = cached_episode_oracle("the answer is 42")
    ok, info = oracle("ignored", "the answer is 42")
    assert ok, info


def test_cached_oracle_normalized_whitespace_match():
    oracle = cached_episode_oracle("  hello   world  ")
    ok, info = oracle("ignored", "hello world")
    assert ok, info


def test_cached_oracle_case_insensitive_match():
    oracle = cached_episode_oracle("Yes")
    ok, info = oracle("ignored", "yes")
    assert ok, info


def test_cached_oracle_strict_numeric_set_match():
    oracle = cached_episode_oracle("7")
    ok, info = oracle("ignored", "7")
    assert ok, info


def test_cached_oracle_rejects_extra_numbers():
    """Verbose wrong answers that happen to contain the right number
    must be rejected. Old heuristic_overlap accepted these."""
    oracle = cached_episode_oracle("7")
    ok, info = oracle("ignored", "answer: 7 because 1 + 2 + 3 + 4 = 10")
    assert not ok
    assert "extra" in info.lower()


def test_cached_oracle_rejects_missing_number():
    oracle = cached_episode_oracle("42")
    ok, info = oracle("ignored", "the answer is 7")
    assert not ok
    assert "missing" in info.lower() or "extra" in info.lower()


def test_cached_oracle_rejects_error_prefix():
    oracle = cached_episode_oracle("42")
    ok, info = oracle("ignored", "Error: not enough numbers")
    assert not ok
    assert "error" in info.lower()


def test_cached_oracle_rejects_empty_expected():
    """No verified ground truth -> never accept (paper invariant: only
    promote a skill against episodes with real solutions)."""
    oracle = cached_episode_oracle("")
    ok, info = oracle("ignored", "anything")
    assert not ok


def test_cached_oracle_rejects_na_expected():
    oracle = cached_episode_oracle("N/A")
    ok, info = oracle("ignored", "anything")
    assert not ok


def test_cached_oracle_negative_numbers():
    oracle = cached_episode_oracle("-5")
    ok, info = oracle("ignored", "-5")
    assert ok, info


def test_cached_oracle_floats_within_tolerance():
    oracle = cached_episode_oracle("3.14159")
    ok, info = oracle("ignored", "3.141590001")
    assert ok, info


def test_cached_oracle_multi_number_match():
    oracle = cached_episode_oracle("answer: 1, 2, 3")
    ok, info = oracle("ignored", "answer: 1, 2, 3")
    assert ok, info


def test_cached_oracle_multi_number_reorder_accepted():
    """Numbers are a set, not a sequence — order should not matter
    when the surface text differs."""
    oracle = cached_episode_oracle("nums: 1 2 3")
    # Same set but reordered surface text. Numeric set match.
    ok, info = oracle("ignored", "the numbers are 3 2 1")
    assert ok, info


def test_cached_oracle_partial_match_rejected():
    """heuristic_overlap accepted >=20% number coverage. The new strict
    oracle must reject when even one expected number is missing."""
    oracle = cached_episode_oracle("1, 2, 3, 4, 5")
    ok, info = oracle("ignored", "1, 2")
    assert not ok


def test_cached_oracle_via_spec():
    spec = {"type": "cached_episode", "expected_solution": "7"}
    oracle = build_oracle_from_spec(spec)
    assert oracle("q", "7")[0]
    assert not oracle("q", "8")[0]


# ---------------------------------------------------------------------------
# Integration with _validate_skill: cached oracle is the new default
# ---------------------------------------------------------------------------


GOOD_STRICT_SKILL = """
def trigger(query: str) -> bool:
    return '+' in query

def execute(query: str) -> str:
    nums = []
    cur = ''
    for c in query + ' ':
        if c.isdigit():
            cur += c
        elif cur:
            nums.append(int(cur))
            cur = ''
    if len(nums) < 2:
        return 'Error: not enough numbers'
    return str(sum(nums))
"""


VERBOSE_WRONG_SKILL = """
def trigger(query: str) -> bool:
    return '+' in query

def execute(query: str) -> str:
    return 'the answer is 7 because 1 + 2 + 3 + 4 = 10'
"""


def _episode(query: str, solution: str, **extra) -> dict:
    ep = {"query": query, "solution": solution, "embedding": [0.0]}
    ep.update(extra)
    return ep


def test_validate_skill_cached_default_accepts_exact():
    """No oracle passed -> cached_episode default kicks in. Strict
    skill that returns exactly the expected number is accepted."""
    episodes = [
        _episode("2 + 5", "7"),
        _episode("3 + 4", "7"),
    ]
    scores, _ = _validate_skill(GOOD_STRICT_SKILL, episodes)
    assert scores["execute_accuracy"] == pytest.approx(1.0)
    assert scores["overall"] >= 0.95


def test_validate_skill_cached_default_rejects_verbose_wrong():
    """A skill whose output happens to contain the right number, with
    extra unrelated numbers, would have been accepted by the legacy
    heuristic_overlap fallback. The new strict default rejects it.
    """
    episodes = [
        _episode("2 + 5", "7"),
        _episode("3 + 4", "7"),
    ]
    scores, report = _validate_skill(VERBOSE_WRONG_SKILL, episodes)
    assert scores["execute_accuracy"] == 0.0
    # Hard gate must trip.
    assert scores["overall"] <= 0.50


def test_validate_skill_legacy_heuristic_opt_in():
    """The old behaviour is still reachable by flipping the config flag,
    so existing users are not silently broken."""
    episodes = [
        _episode("2 + 5", "7"),
        _episode("3 + 4", "7"),
    ]
    legacy_cfg = replace(
        DEFAULT_CONFIG,
        skill_validation=replace(
            DEFAULT_CONFIG.skill_validation,
            use_heuristic_overlap_fallback=True,
        ),
    )
    # Verbose-wrong with legacy heuristic: '7' is in the candidate, so
    # it will pass the heuristic overlap check.
    scores, _ = _validate_skill(VERBOSE_WRONG_SKILL, episodes, config=legacy_cfg)
    # Legacy is leaky — at least one execute pass should happen.
    assert scores["execute_accuracy"] > 0.0
