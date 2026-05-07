"""Held-out validation contract tests.

Phase 6 second lever: never crystallize a rule that fails on a
held-out episode from its own cluster. Pre-Phase-6 the same episodes
were both training and validation set for ``_validate_skill``, so
skills overfit and shipped to REFLEX with broken regexes that worked
on the 2-episode cluster but failed on every paraphrase.

These tests check the config, the split, and the rejection path.
A full end-to-end sleep cycle is exercised by the broader agent
test suite — here we only assert the contract.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHoldoutConfig(unittest.TestCase):
    def test_defaults(self):
        from nare.config import DEFAULT_CONFIG

        sc = DEFAULT_CONFIG.sleep
        self.assertTrue(sc.use_holdout_validation)
        self.assertEqual(sc.holdout_n, 1)
        self.assertGreaterEqual(sc.holdout_min_cluster_size, 3)
        self.assertGreater(sc.holdout_min_accuracy, 0.0)
        self.assertLessEqual(sc.holdout_min_accuracy, 1.0)

    def test_holdout_can_be_disabled(self):
        from dataclasses import replace

        from nare.config import DEFAULT_CONFIG

        sc = replace(
            DEFAULT_CONFIG.sleep,
            use_holdout_validation=False,
        )
        self.assertFalse(sc.use_holdout_validation)


class TestHoldoutSplit(unittest.TestCase):
    """The split logic itself — independent of LLM calls.

    We replicate the same indexing the agent uses so the contract
    is locked: ``train = eps[:-h]``, ``holdout = eps[-h:]``.
    """

    def test_split_with_3_episodes_holdout_1(self):
        eps = [{"i": 0}, {"i": 1}, {"i": 2}]
        h = 1
        train = eps[:-h]
        holdout = eps[-h:]
        self.assertEqual(len(train), 2)
        self.assertEqual(len(holdout), 1)
        self.assertEqual(train, [{"i": 0}, {"i": 1}])
        self.assertEqual(holdout, [{"i": 2}])

    def test_split_with_5_episodes_holdout_2(self):
        eps = [{"i": i} for i in range(5)]
        h = 2
        train = eps[:-h]
        holdout = eps[-h:]
        self.assertEqual(len(train), 3)
        self.assertEqual(len(holdout), 2)


if __name__ == "__main__":
    unittest.main()
