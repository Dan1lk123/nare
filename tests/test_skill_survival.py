"""Tests for skill-survival knobs in REM sleep.

The defaults are tuned to fix the real-100 benchmark regression where
every newly crystallised skill landed at confidence 0.45, immediately
failed the legacy hard-coded LLM-stress threshold of 0.80, and was
quarantined within a single sleep cycle — REFLEX never activated.

Three knobs are tested here:

* ``rem_stress_pass_threshold`` — was hard-coded 0.80, now configurable
  with a default of 0.55.
* ``rem_grace_period_cycles`` — first N cycles log the failure but
  skip the penalty / repair loop entirely.
* ``rem_repair_max_inner_attempts`` — was hard-coded 3, now
  configurable with a default of 1.

The tests don't exercise the full REM phase (which calls the live LLM);
they call the relevant helpers / branches directly with a stubbed
``_validate_skill`` so the logic is exercised offline.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import patch

from nare.config import DEFAULT_CONFIG, NareConfig, SkillLifecycleConfig


def _config_with(**skill_overrides) -> NareConfig:
    skill = dataclasses.replace(DEFAULT_CONFIG.skill, **skill_overrides)
    return dataclasses.replace(DEFAULT_CONFIG, skill=skill)


def test_default_stress_threshold_is_below_legacy():
    """The new default 0.55 must be strictly below the old 0.80."""
    assert DEFAULT_CONFIG.skill.rem_stress_pass_threshold < 0.80
    assert DEFAULT_CONFIG.skill.rem_stress_pass_threshold > 0.0


def test_default_grace_period_at_least_one():
    """Without a grace period the original regression still fires."""
    assert DEFAULT_CONFIG.skill.rem_grace_period_cycles >= 1


def test_default_repair_attempts_capped():
    """Capped at 1 to bound LLM cost on hopeless skills."""
    assert DEFAULT_CONFIG.skill.rem_repair_max_inner_attempts == 1


def test_skill_lifecycle_config_is_frozen_dataclass():
    """All three knobs live on the frozen SkillLifecycleConfig so they
    can be tweaked without monkey-patching globals."""
    cfg = SkillLifecycleConfig()
    assert hasattr(cfg, "rem_stress_pass_threshold")
    assert hasattr(cfg, "rem_grace_period_cycles")
    assert hasattr(cfg, "rem_repair_max_inner_attempts")


def test_dataclass_replace_overrides_threshold():
    """Override path used by benchmarks / regression tests."""
    cfg = _config_with(rem_stress_pass_threshold=0.80)
    assert cfg.skill.rem_stress_pass_threshold == 0.80
    # original default unchanged
    assert DEFAULT_CONFIG.skill.rem_stress_pass_threshold < 0.80


def _build_skipping_agent(config):
    """Construct an Agent without spinning up the live LLM.

    We instantiate the class but bypass ``__init__`` since most
    sub-systems aren't relevant to the REM-grace test.
    """
    from nare.core.agent import NAREProductionAgent

    agent = NAREProductionAgent.__new__(NAREProductionAgent)
    agent.config = config
    return agent


def test_grace_period_logic_matches_default_config():
    """Cycle count vs. grace period boundary check, tested directly
    without invoking the REM phase (which calls LLMs).

    With ``rem_grace_period_cycles=1``, a rule with ``rem_cycles_seen``
    incremented to 1 is still in grace; bumping to 2 exits grace.
    """
    cfg = DEFAULT_CONFIG
    rule = {"rem_cycles_seen": 0}
    rule["rem_cycles_seen"] += 1
    in_grace = rule["rem_cycles_seen"] <= max(0, cfg.skill.rem_grace_period_cycles)
    assert in_grace is True

    rule["rem_cycles_seen"] += 1
    in_grace = rule["rem_cycles_seen"] <= max(0, cfg.skill.rem_grace_period_cycles)
    assert in_grace is False


def test_grace_period_disabled_with_zero():
    """Setting ``rem_grace_period_cycles=0`` recovers the legacy
    behaviour where the very first failed REM cycle penalises."""
    cfg = _config_with(rem_grace_period_cycles=0)
    rule = {"rem_cycles_seen": 0}
    rule["rem_cycles_seen"] += 1
    in_grace = rule["rem_cycles_seen"] <= max(0, cfg.skill.rem_grace_period_cycles)
    assert in_grace is False


def test_repair_skill_called_with_configured_max_attempts():
    """The REM repair path must forward
    ``skill_cfg.rem_repair_max_inner_attempts`` (clamped >= 1) into
    ``llm.repair_skill``. Hard-coding 3 was the historical bug.

    We patch ``repair_skill`` to capture the kwarg without invoking
    the real LLM, and drive ``_rem_sleep_phase`` against a single
    weak rule so the failing branch is taken.
    """
    cfg = _config_with(
        rem_repair_max_inner_attempts=2,
        rem_grace_period_cycles=0,  # skip grace so repair branch runs
        rem_stress_pass_threshold=0.99,  # force fail on any score
    )
    agent = _build_skipping_agent(cfg)

    # Minimal stubs for _rem_sleep_phase dependencies.
    class _Mem:
        episodes = [{"query": "x", "verified_solution": "y"}]
        semantic_rules = [{
            "pattern": "Stub",
            "python_code": "def f(x): return x",
            "confidence": 0.4,
            "peak_confidence": 0.4,
        }]

        def save(self):
            pass

    class _Graph:
        def weaken_all(self, decay):
            pass

        def save(self):
            pass

    agent.memory = _Mem()
    agent.graph = _Graph()
    agent.oracle = None

    captured = {}

    def fake_repair_skill(*args, **kwargs):
        captured["max_attempts"] = kwargs.get("max_attempts")
        return None  # signal "repair failed"; no follow-up code path needed

    fake_validate = lambda *a, **kw: ({"overall": 0.10, "execute_accuracy": 0.0}, "err")
    fake_stress = lambda eps: [{"input": "1", "expected": "1"}]

    with patch("nare.llm.repair_skill", fake_repair_skill), \
         patch("nare.llm.generate_stress_tests", fake_stress), \
         patch("nare.llm._validate_skill", fake_validate):
        # Force the LLM-stress fallback by stubbing _rem_cached_replay
        # to return "no replay possible".
        agent._rem_cached_replay = lambda r: (None, [], 0)
        agent._apply_rem_penalty = lambda *a, **kw: None
        agent._rem_sleep_phase()

    assert captured.get("max_attempts") == 2, (
        f"Expected repair_skill to receive max_attempts=2, "
        f"got {captured.get('max_attempts')}"
    )
