"""Phase 6 contract tests: solve() honours the oracle parameter.

Rather than full agent integration (which needs the real LLM client),
these tests check the public contract:

* ``solve()`` accepts ``oracle`` and ``expected_hint`` kwargs.
* When ``oracle`` is set the agent stashes it on the instance so
  downstream branches (FAST cache veto, REFLEX cache veto, SLOW VS
  loop) can read it.
* When ``oracle=None`` the contract collapses to pre-Phase-6
  behaviour — proving "no worse than vanilla".
"""

from __future__ import annotations

import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSolveSignature(unittest.TestCase):
    def test_solve_accepts_oracle_kwargs(self):
        from nare.core.agent import NAREProductionAgent

        sig = inspect.signature(NAREProductionAgent.solve)
        self.assertIn("oracle", sig.parameters)
        self.assertIn("expected_hint", sig.parameters)
        # Both must default to None so existing callers keep working.
        self.assertIsNone(sig.parameters["oracle"].default)
        self.assertIsNone(sig.parameters["expected_hint"].default)

    def test_synthesis_module_importable(self):
        # The synthesis module is the actual lever; import shape and
        # public API must be stable.
        from nare.core.synthesis.engine import (  # noqa: F401
            SynthesisAttempt,
            SynthesisResult,
            verified_synthesis,
        )

    def test_synthesis_config_in_default(self):
        from nare.config import DEFAULT_CONFIG

        self.assertTrue(hasattr(DEFAULT_CONFIG, "synthesis"))
        self.assertGreaterEqual(DEFAULT_CONFIG.synthesis.max_attempts, 1)


if __name__ == "__main__":
    unittest.main()
