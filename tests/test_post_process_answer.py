"""Regression tests for ``NAREProductionAgent._post_process_answer``.

The user's last benchmark log showed every path (FAST, HYBRID, SLOW)
echoing back fenced ```python``` blocks instead of the executed value
on Tasks 5 / 14 / 15 / 18 / 20 / 21 / 22 — regressing accuracy from
91.7% to 83.3%. The fix is a single helper that extracts and runs the
fenced block in the AST sandbox, with conservative fallbacks on every
failure mode.

These tests exercise the helper directly (no LLM, no FAISS, no real
agent state) so they're cheap and deterministic.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nare.core.agent import NAREProductionAgent  # noqa: E402


def _fresh_agent_helper():
    """Bind only the helper without booting the real agent (which loads
    embeddings, FAISS indices, neural memory, etc)."""
    agent = MagicMock(spec=NAREProductionAgent)
    agent._post_process_answer = NAREProductionAgent._post_process_answer.__get__(
        agent, NAREProductionAgent
    )
    return agent


class TestPostProcessAnswer(unittest.TestCase):
    def test_returns_plain_text_unchanged(self):
        a = _fresh_agent_helper()
        log = []
        out = a._post_process_answer("42", "FAST", log)
        self.assertEqual(out, "42")
        self.assertEqual(log, [])

    def test_returns_empty_unchanged(self):
        a = _fresh_agent_helper()
        log = []
        self.assertEqual(a._post_process_answer("", "FAST", log), "")
        self.assertEqual(a._post_process_answer(None, "FAST", log), None)

    def test_executes_fenced_python_block(self):
        a = _fresh_agent_helper()
        log = []
        text = (
            "Mathematical computation. Exponentiation.\n\n"
            "```python\n"
            "result = 2 ** 10\n"
            "print(result)\n"
            "```"
        )
        out = a._post_process_answer(text, "SLOW", log)
        self.assertEqual(out, "1024")
        self.assertTrue(any("Executed inline code block" in m for m in log))

    def test_executes_email_extraction_block(self):
        """The exact code shape from Task 6/7/8 of the benchmark."""
        a = _fresh_agent_helper()
        log = []
        text = (
            "```python\n"
            "import re\n"
            "text = 'Send your resume to hr@jobs.io and a copy to manager@jobs.io. Questions? help@jobs.io'\n"
            "emails = re.findall(r'[\\w.+-]+@[\\w-]+\\.[\\w.-]+', text)\n"
            "print(emails)\n"
            "```"
        )
        out = a._post_process_answer(text, "HYBRID", log)
        self.assertIn("hr@jobs.io", out)
        self.assertIn("manager@jobs.io", out)

    def test_falls_back_when_block_fails_silently(self):
        """A block that prints nothing and has no `result` returns the original."""
        a = _fresh_agent_helper()
        log = []
        text = "Some explanation.\n```python\nx = 1 + 1\n```"
        out = a._post_process_answer(text, "SLOW", log)
        # Falls back to original text, with a log line about no usable output
        self.assertEqual(out, text)
        self.assertTrue(any("no usable output" in m for m in log))

    def test_falls_back_when_block_blocked_by_sandbox(self):
        """A block doing forbidden ops returns original text + sandbox log."""
        a = _fresh_agent_helper()
        log = []
        text = "```python\nimport os\nos.system('ls')\n```"
        out = a._post_process_answer(text, "SLOW", log)
        # Sandbox should reject — original kept.
        self.assertEqual(out, text)
        self.assertTrue(
            any("blocked by sandbox" in m or "raised" in m for m in log)
        )

    def test_uses_result_variable_when_no_print(self):
        a = _fresh_agent_helper()
        log = []
        text = "```python\nresult = 5040\n```"
        out = a._post_process_answer(text, "FAST", log)
        self.assertEqual(out, "5040")

    def test_returns_original_for_non_python_fence(self):
        a = _fresh_agent_helper()
        log = []
        text = "Here is some output:\n```\nrandom text\n```"
        out = a._post_process_answer(text, "FAST", log)
        self.assertEqual(out, text)


if __name__ == "__main__":
    unittest.main()
