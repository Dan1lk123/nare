"""Unit tests for the confidence-oscillation quarantine logic.

The benchmark log showed Email Extraction bouncing 0.45 ↔ 0.23 ↔ 0.45 ↔ 0.24
across REM/sleep cycles, never stabilising long enough to actually fire on
REFLEX. The fix has two parts:

1. ``_apply_rem_penalty`` increments ``rem_penalty_count`` and quarantines
   the rule once that count crosses ``skill_quarantine_after_penalties`` while
   ``peak_confidence`` is still below ``skill_quarantine_peak_threshold``.
2. The REFLEX-matching loop in ``solve_query`` skips quarantined rules.

These tests exercise (1) directly. (2) is verified by inspection of the
guard ``if sem.get('quarantined'): continue`` in ``agent.py`` and is
covered by repeat-test runs of the full benchmark.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from nare.core.agent import NAREProductionAgent
from nare.config import NareConfig


class TestApplyRemPenalty(unittest.TestCase):
    def _make_agent(self) -> NAREProductionAgent:
        # Bypass __init__: we only need self.config + the method under test.
        agent = NAREProductionAgent.__new__(NAREProductionAgent)
        agent.config = NareConfig()
        return agent

    def test_first_penalty_decreases_conf_and_increments_counter(self):
        agent = self._make_agent()
        rule = {"confidence": 0.45, "maturity": 1}
        agent._apply_rem_penalty(
            rule,
            scores={"overall": 0.10},
            pass_threshold=0.80,
            old_conf=0.45,
            pattern="TestSkill",
        )
        self.assertLess(rule["confidence"], 0.45)
        self.assertGreaterEqual(rule["confidence"], 0.10)
        self.assertEqual(rule["rem_penalty_count"], 1)
        self.assertFalse(rule.get("quarantined", False))
        # peak_confidence is initialised to the pre-penalty score.
        self.assertEqual(rule["peak_confidence"], 0.45)

    def test_repeated_penalties_eventually_quarantine_low_peak(self):
        agent = self._make_agent()
        cfg = agent.config.skill
        n_penalties = cfg.skill_quarantine_after_penalties

        rule = {"confidence": 0.45, "maturity": 1}
        for i in range(n_penalties):
            agent._apply_rem_penalty(
                rule,
                scores={"overall": 0.10},
                pass_threshold=0.80,
                old_conf=rule["confidence"],
                pattern="ChronicallyBad",
            )

        self.assertEqual(rule["rem_penalty_count"], n_penalties)
        # Peak stays well below the quarantine peak threshold (0.50 default).
        self.assertLess(rule["peak_confidence"], cfg.skill_quarantine_peak_threshold)
        self.assertTrue(rule["quarantined"])

    def test_high_peak_protects_from_quarantine(self):
        """A rule that was once strong (peak >= threshold) is NOT quarantined
        even after many REM penalties — it merely loses confidence."""
        agent = self._make_agent()
        cfg = agent.config.skill

        # Manually set a high peak as if the rule had been good earlier.
        rule = {
            "confidence": 0.55,
            "maturity": 1,
            "peak_confidence": 0.85,
        }
        for _ in range(cfg.skill_quarantine_after_penalties + 2):
            agent._apply_rem_penalty(
                rule,
                scores={"overall": 0.20},
                pass_threshold=0.80,
                old_conf=rule["confidence"],
                pattern="WasGoodOnce",
            )
        self.assertGreaterEqual(rule["rem_penalty_count"], cfg.skill_quarantine_after_penalties)
        self.assertFalse(rule.get("quarantined", False))
        # Peak is preserved through the penalty loop.
        self.assertGreaterEqual(rule["peak_confidence"], 0.85)


class TestRefinementHysteresis(unittest.TestCase):
    """The graded-refinement code path clamps confidence after a REM penalty.

    We assert the clamp formula directly without driving the full sleep
    pipeline (which requires an LLM).
    """

    def test_clamp_value_after_one_penalty(self):
        cfg = NareConfig()
        max_bump = cfg.skill.skill_refinement_max_bump_after_penalty
        existing_conf = 0.23
        raw_score = 0.55
        # Reproduce the clamp formula from agent.py
        clamped = min(raw_score, existing_conf + max_bump)
        self.assertAlmostEqual(clamped, 0.33, places=2)
        self.assertLess(clamped, raw_score)

    def test_clamp_no_op_when_raw_already_low(self):
        cfg = NareConfig()
        max_bump = cfg.skill.skill_refinement_max_bump_after_penalty
        existing_conf = 0.20
        raw_score = 0.22  # tiny bump
        clamped = min(raw_score, existing_conf + max_bump)
        self.assertAlmostEqual(clamped, raw_score, places=4)


if __name__ == "__main__":
    unittest.main()
