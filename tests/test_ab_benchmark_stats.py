"""Tests for the A/B benchmark's statistical helpers.

These check the parts of ``benchmarks/ab_benchmark.py`` that don't
depend on an LLM — Wilson interval and paired-bootstrap CI — so we
can guarantee the headline numbers in the user's report are computed
honestly.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

from benchmarks.ab_benchmark import (  # noqa: E402
    paired_bootstrap_ci,
    run_paired,
    wilson_ci,
)


class TestWilsonCI(unittest.TestCase):
    def test_extremes(self):
        # 0/0 → conservative [0, 1]
        lo, hi = wilson_ci(0, 0)
        self.assertEqual(lo, 0.0)
        self.assertEqual(hi, 1.0)

    def test_perfect_score_has_finite_lower_bound(self):
        # 5/5 → upper bound ≤ 1, lower bound > 0 (the whole point of
        # Wilson vs naive proportion).
        lo, hi = wilson_ci(5, 5)
        self.assertGreater(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_zero_score_has_finite_upper_bound(self):
        lo, hi = wilson_ci(0, 5)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)
        self.assertLess(hi, 1.0)

    def test_known_value(self):
        # 14/15 should give roughly [70%, 99%] which matches the user's
        # external report.
        lo, hi = wilson_ci(14, 15)
        self.assertGreater(lo, 0.65)
        self.assertLess(lo, 0.75)
        self.assertGreater(hi, 0.95)


class TestPairedBootstrapCI(unittest.TestCase):
    def test_identical_arms_includes_zero(self):
        # Both arms identical → CI must include 0.
        pairs = [(True, True)] * 10 + [(False, False)] * 5
        lo, hi = paired_bootstrap_ci(pairs, n_boot=2000, rng_seed=42)
        self.assertLessEqual(lo, 0.0)
        self.assertGreaterEqual(hi, 0.0)

    def test_strictly_better_arm_has_positive_ci(self):
        # NARE strictly beats vanilla on every paired draw.
        pairs = [(True, False)] * 30
        lo, hi = paired_bootstrap_ci(pairs, n_boot=2000, rng_seed=42)
        self.assertGreater(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_strictly_worse_arm_has_negative_ci(self):
        pairs = [(False, True)] * 30
        lo, hi = paired_bootstrap_ci(pairs, n_boot=2000, rng_seed=42)
        self.assertLess(lo, 0.0)
        self.assertLess(hi, 0.0)

    def test_seed_determinism(self):
        pairs = [(True, False), (False, True), (True, True), (False, False)] * 8
        a = paired_bootstrap_ci(pairs, n_boot=1000, rng_seed=7)
        b = paired_bootstrap_ci(pairs, n_boot=1000, rng_seed=7)
        self.assertEqual(a, b)


class TestRunPaired(unittest.TestCase):
    """The shared-attempt-1 contract is: vanilla and NARE must see
    the *same* attempt-1 raw response.  We verify by stubbing the LLM
    so attempt-1 returns a fixed string, retries return a different
    fixed string, and checking that vanilla used the attempt-1 string
    while NARE either matched it (oracle accepted) or upgraded via a
    retry (oracle rejected attempt-1, retry produced a passing
    answer).
    """

    def setUp(self):
        # Patch llm.generate_samples on the module imported by the
        # harness so both arms use our stubs.
        from benchmarks import ab_benchmark
        self.ab = ab_benchmark
        self._real_gen = ab_benchmark.llm.generate_samples
        self._call_count = {"n": 0}

    def tearDown(self):
        self.ab.llm.generate_samples = self._real_gen

    def test_vanilla_sees_shared_attempt1(self):
        """When attempt-1 already gives a correct answer, vanilla
        and NARE must produce the *exact same* answer (no extra LLM
        calls happened on NARE side)."""

        def fake_gen(prompt, n=1, temperature=0.0, mode="SLOW"):
            self._call_count["n"] += 1
            return [{"solution": "```python\nprint(42)\n```"}], 0

        self.ab.llm.generate_samples = fake_gen
        task = {
            "id": "t1", "category": "test",
            "query": "Compute 42.",
            "oracle_spec": {"type": "numeric_set", "expected": [42]},
        }
        nare, vanilla = self.ab.run_paired(task)
        self.assertEqual(nare.answer, vanilla.answer)
        self.assertTrue(nare.correct)
        self.assertTrue(vanilla.correct)
        # Exactly one LLM call (the shared attempt-1) — VS short-
        # circuited because oracle passed on the seeded attempt 1.
        self.assertEqual(self._call_count["n"], 1)

    def test_nare_can_upgrade_via_retry(self):
        """When attempt-1 is wrong, vanilla ships the wrong answer
        but NARE's VS retry produces a correct one."""
        responses = [
            "```python\nprint(0)\n```",   # attempt 1 — wrong (oracle: missing 42)
            "```python\nprint(42)\n```",  # retry — correct
        ]

        def fake_gen(prompt, n=1, temperature=0.0, mode="SLOW"):
            i = self._call_count["n"]
            self._call_count["n"] += 1
            return [{"solution": responses[min(i, len(responses) - 1)]}], 0

        self.ab.llm.generate_samples = fake_gen
        task = {
            "id": "t2", "category": "test",
            "query": "Compute 42.",
            "oracle_spec": {"type": "numeric_set", "expected": [42]},
        }
        nare, vanilla = self.ab.run_paired(task)
        self.assertFalse(vanilla.correct)
        self.assertTrue(nare.correct)
        # Retry called the LLM at least once more after attempt 1.
        self.assertGreaterEqual(self._call_count["n"], 2)

    def test_nare_never_worse_than_vanilla_when_retries_drift(self):
        """Hard contract: even if every retry gives wrong answers,
        NARE must not be MORE wrong than vanilla — both share the
        same attempt 1, so worst case both are equal."""
        responses = [
            "```python\nprint(0)\n```",   # attempt 1 — wrong
            "```python\nprint(99)\n```",  # retry — also wrong
            "```python\nprint(99)\n```",
            "```python\nprint(99)\n```",
            "```python\nprint(99)\n```",
        ]

        def fake_gen(prompt, n=1, temperature=0.0, mode="SLOW"):
            i = self._call_count["n"]
            self._call_count["n"] += 1
            return [{"solution": responses[min(i, len(responses) - 1)]}], 0

        self.ab.llm.generate_samples = fake_gen
        task = {
            "id": "t3", "category": "test",
            "query": "Compute 42.",
            "oracle_spec": {"type": "numeric_set", "expected": [42]},
        }
        nare, vanilla = self.ab.run_paired(task)
        # Both wrong, never NARE worse than vanilla.
        self.assertFalse(vanilla.correct)
        self.assertFalse(nare.correct)
        # The NARE bound: should fall back to attempt-1 output,
        # equal to vanilla's answer ("0").
        self.assertEqual(nare.answer, "0")
        self.assertEqual(vanilla.answer, "0")


if __name__ == "__main__":
    unittest.main()
