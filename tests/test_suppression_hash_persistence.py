"""Regression tests for the immune-system suppression hash fix.

The bug (Devin Review on commit 436b060):
  ``add_suppression_rule`` stored Python ``hash()`` ints in
  ``query_hash`` / ``answer_hash``. ``hash()`` is randomised per-process
  via PYTHONHASHSEED (CPython 3.3+), so saved rules silently stopped
  matching after the next process start — every suppression rule
  became dead on restart and the immune system's blocking mechanism
  was non-functional across save/load cycles.

The fix:
  Use ``MemorySystem._suppression_hash`` (SHA1 → 16 hex chars). This
  test exercises save → reload → ``is_suppressed`` round-trip and
  also covers in-process matching.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nare.config import DEFAULT_CONFIG  # noqa: E402
from nare.memory import MemorySystem  # noqa: E402


class TestSuppressionHashPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fresh_memory(self) -> MemorySystem:
        return MemorySystem(
            embedding_dim=32,
            persist_dir=self.tmp,
            config=DEFAULT_CONFIG,
        )

    def test_in_process_match(self):
        mem = self._fresh_memory()
        emb = np.zeros(32, dtype=np.float32)
        mem.add_suppression_rule("Bad query", "Bad answer", emb)
        self.assertTrue(mem.is_suppressed("Bad query", "Bad answer"))
        self.assertTrue(mem.is_suppressed("BAD QUERY", "Bad answer"))  # case-insens
        self.assertFalse(mem.is_suppressed("Different", "Bad answer"))

    def test_match_survives_save_reload(self):
        """Open mem → suppress → drop instance → reopen → still suppressed."""
        mem1 = self._fresh_memory()
        emb = np.zeros(32, dtype=np.float32)
        mem1.add_suppression_rule("Toxic Q", "Toxic A", emb)
        # Drop instance so any per-process state goes away.
        del mem1

        mem2 = self._fresh_memory()
        # Sanity: rule was actually persisted to disk.
        self.assertTrue(
            mem2.suppression_rules,
            "suppression rule did not persist to disk",
        )
        # The actual regression: after reload, the rule must still match.
        self.assertTrue(mem2.is_suppressed("Toxic Q", "Toxic A"))

    def test_hash_is_string_not_int(self):
        """Catch a future regression that swaps the hash back to ``hash()``."""
        mem = self._fresh_memory()
        emb = np.zeros(32, dtype=np.float32)
        mem.add_suppression_rule("Bad query", "Bad answer", emb)
        rule = mem.suppression_rules[0]
        self.assertIsInstance(rule["query_hash"], str)
        self.assertIsInstance(rule["answer_hash"], str)
        # 16 hex chars per _suppression_hash convention.
        self.assertEqual(len(rule["query_hash"]), 16)
        self.assertEqual(len(rule["answer_hash"]), 16)
        # And it had better be the SHA1 prefix, not a hexlified int.
        int_attempt = format(abs(hash("bad query")), "x")[:16]
        self.assertNotEqual(rule["query_hash"], int_attempt)


if __name__ == "__main__":
    unittest.main()
