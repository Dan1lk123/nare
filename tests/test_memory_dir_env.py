"""Regression test for ``NARE_MEMORY_DIR`` env-var contract.

Devin Review on commit ac8ab52:
   ``benchmarks/full_benchmark.py`` and ``benchmarks/quick_test.py``
   both set ``os.environ["NARE_MEMORY_DIR"]`` before constructing the
   agent, but ``MemorySystem.__init__`` previously hard-coded
   ``persist_dir="memory_store"`` and never read the env var. Memory
   isolation between benchmark runs and the global store was silently
   broken — every benchmark contaminated ``./memory_store/`` and was
   contaminated *by* it.

Fixed in the same commit by reading ``NARE_MEMORY_DIR`` as a default
in ``MemorySystem.__init__`` when ``persist_dir`` is None. This test
locks that contract.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nare.config import DEFAULT_CONFIG  # noqa: E402
from nare.memory import MemorySystem  # noqa: E402


class TestMemoryDirEnv(unittest.TestCase):
    def setUp(self):
        self.prev_env = os.environ.get("NARE_MEMORY_DIR")
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        if self.prev_env is None:
            os.environ.pop("NARE_MEMORY_DIR", None)
        else:
            os.environ["NARE_MEMORY_DIR"] = self.prev_env
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_env_var_picked_up_when_no_arg(self):
        """No explicit persist_dir → env var wins over default."""
        os.environ["NARE_MEMORY_DIR"] = self.tmp
        mem = MemorySystem(embedding_dim=32, config=DEFAULT_CONFIG)
        self.assertEqual(mem.persist_dir, self.tmp)
        # Directory is created at construction time, files arrive on
        # the first ``save()``. Check both contracts.
        self.assertTrue(os.path.isdir(self.tmp))
        mem.save()
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "episodic.faiss")))

    def test_explicit_arg_beats_env_var(self):
        """Explicit persist_dir argument has highest priority."""
        os.environ["NARE_MEMORY_DIR"] = "/should/not/be/used"
        explicit = self.tmp
        mem = MemorySystem(
            embedding_dim=32,
            persist_dir=explicit,
            config=DEFAULT_CONFIG,
        )
        self.assertEqual(mem.persist_dir, explicit)

    def test_default_when_neither_set(self):
        """No arg, no env var → default ``memory_store``."""
        os.environ.pop("NARE_MEMORY_DIR", None)
        # Sandbox into tmp so we don't pollute CWD.
        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            mem = MemorySystem(embedding_dim=32, config=DEFAULT_CONFIG)
            self.assertEqual(mem.persist_dir, "memory_store")
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
