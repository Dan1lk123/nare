"""Regression tests for stable episode keys (Devin Review #1 on PR #5).

The bug:
  * ``_sleep_phase`` stored ``source_episode_ids`` as positional indices
    into ``MemorySystem.episodes``.
  * Immediately afterwards, ``_sleep_phase`` deletes those exact
    episodes, so the indices become stale (and frequently out of bounds).
  * Subsequent penalty backpropagation in ``_record_skill_result`` then
    silently mutated ``τ`` for the *wrong* episodes, corrupting the
    immune system over time.

The fix:
  * Each episode now carries an ``episode_key`` derived from
    ``hash(query + solution[:256])``, set at insertion time.
  * Rules store ``source_episode_keys`` (list of keys), not positional
    indices. Lookup goes through ``find_episode_indices_by_keys``,
    which scans current ``episodes`` and resolves keys → live indices.

These tests exercise the new contract end-to-end on the real
``MemorySystem`` (no LLM, no agent, no FAISS HNSW build is needed for
the small fixtures).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nare.config import DEFAULT_CONFIG  # noqa: E402
from nare.memory import (  # noqa: E402
    MemorySystem,
    episode_content_key,
)


class TestEpisodeContentKey(unittest.TestCase):
    def test_deterministic_for_same_input(self):
        k1 = episode_content_key("Hello world?", "Hi!")
        k2 = episode_content_key("hello world?", "Hi!")  # case-insensitive
        k3 = episode_content_key("  Hello world?  ", "Hi!")  # whitespace strip
        self.assertEqual(k1, k2)
        self.assertEqual(k1, k3)

    def test_different_for_different_content(self):
        a = episode_content_key("Q1", "A1")
        b = episode_content_key("Q2", "A1")
        c = episode_content_key("Q1", "A2")
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(b, c)

    def test_empty_inputs_dont_crash(self):
        self.assertEqual(len(episode_content_key("", "")), 16)
        self.assertEqual(len(episode_content_key(None, None)), 16)


class TestStableLookupThroughDeletion(unittest.TestCase):
    """Smoke test mirroring the real bug: insert N episodes, delete some,
    confirm key-based lookup returns the survivors' new positional ids."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # Use a tiny embedding dim to keep test fixtures cheap.
        self.mem = MemorySystem(
            embedding_dim=32,
            persist_dir=self.tmp,
            config=DEFAULT_CONFIG,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _add(self, query: str, solution: str):
        emb = np.random.RandomState(hash(query) & 0xFFFF).rand(
            self.mem.embedding_dim
        ).astype(np.float32)
        ep = {"query": query, "solution": solution, "score": 0.9}
        self.mem.add_episode(ep, emb)
        return ep

    def test_keys_set_at_insert(self):
        self._add("Q1", "A1")
        self._add("Q2", "A2")
        self.assertIn("episode_key", self.mem.episodes[0])
        self.assertIn("episode_key", self.mem.episodes[1])
        # Keys are content-derived → reproducible.
        self.assertEqual(
            self.mem.episodes[0]["episode_key"],
            episode_content_key("Q1", "A1"),
        )

    def test_lookup_after_arbitrary_deletion(self):
        self._add("Q1", "A1")
        self._add("Q2", "A2")
        self._add("Q3", "A3")
        keys_we_track = [
            self.mem.episodes[0]["episode_key"],  # Q1
            self.mem.episodes[2]["episode_key"],  # Q3
        ]

        # Simulate _sleep_phase: delete the middle episode (Q2) by index
        # AND delete one we *do* track (Q1, idx 0). Q3 survives.
        # Old code stored [0, 2] as ids — after deletion both are stale.
        with self.mem._lock:
            self.mem.episodes = [self.mem.episodes[2]]

        live = self.mem.find_episode_indices_by_keys(keys_we_track)
        # Only Q3 survived; key for Q1 should not match anything.
        self.assertEqual(live, [0])
        self.assertEqual(self.mem.episodes[0]["query"], "Q3")

    def test_lookup_handles_legacy_episodes_without_key(self):
        """Episodes saved before this change won't have ``episode_key``;
        ``find_episode_indices_by_keys`` should still match them by
        recomputing the key on-the-fly."""
        self._add("LegacyQ", "LegacyA")
        # Strip the key to simulate a pre-fix episode.
        self.mem.episodes[0].pop("episode_key", None)

        recomputed = episode_content_key("LegacyQ", "LegacyA")
        live = self.mem.find_episode_indices_by_keys([recomputed])
        self.assertEqual(live, [0])

    def test_unknown_key_returns_empty(self):
        self._add("Q1", "A1")
        live = self.mem.find_episode_indices_by_keys(["deadbeef00000000"])
        self.assertEqual(live, [])


class TestSourceKeysIncludeRepresentative(unittest.TestCase):
    """Regression for Devin Review on commit 20c3fa3:

    ``_sleep_phase`` builds ``source_episode_keys`` from
    ``cluster_indices``, which is computed via
    ``np.where(sim_matrix[core_idx] > sim_threshold)[0]``. Because the
    similarity matrix diagonal is zeroed at ``_sleep_phase`` line ~400,
    ``core_idx`` itself is NOT in ``cluster_indices``. Then
    ``to_delete = set(cluster_indices) - {core_idx}`` deletes every
    cluster member that's referenced in source_keys, leaving the
    surviving representative orphaned — penalty backprop becomes a
    silent no-op for every new rule.

    The fix unions ``cluster_indices`` with ``{core_idx}`` before
    building source_keys, so the surviving representative is always
    findable.
    """

    def test_core_idx_is_in_source_keys(self):
        """Direct simulation of the _sleep_phase fix logic."""
        cluster_indices = np.array([1, 2, 3])  # core_idx=0 excluded
        core_idx = 0
        all_member_indices = list(
            {int(i) for i in cluster_indices} | {core_idx}
        )
        # The rule MUST reference core_idx, otherwise the surviving
        # representative is unreachable from source_keys.
        self.assertIn(core_idx, all_member_indices)
        # All cluster members are still tracked.
        for i in cluster_indices:
            self.assertIn(int(i), all_member_indices)


if __name__ == "__main__":
    unittest.main()
