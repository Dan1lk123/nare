"""Tests for nare.synthesis.verified_synthesis.

These tests use stub ``propose_fn`` callables and synthetic oracles
so they require neither an LLM nor a network connection.

The contract under test:

1. With no oracle, the loop returns attempt 1 verbatim — strictly no
   worse than vanilla CoT.
2. With an oracle, the loop converges on the first oracle-passing
   attempt.
3. Oracle diagnostic info is passed back to ``propose_fn`` as part of
   the feedback prompt so subsequent attempts can self-correct.
4. Code-block extraction + sandbox execution happen automatically;
   plain-text responses are passed through.
5. Sandbox errors do not crash the loop — they are recorded as
   ``attempt.error`` and the loop keeps going.
6. ``max_attempts`` is honoured.
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nare.core.synthesis.engine import (  # noqa: E402
    SynthesisAttempt,
    SynthesisResult,
    verified_synthesis,
)


class TestVerifiedSynthesis(unittest.TestCase):
    # ------------------------------------------------------------------
    # No-oracle baseline
    # ------------------------------------------------------------------

    def test_no_oracle_single_attempt(self):
        calls = {"n": 0}

        def propose(prompt, prior):
            calls["n"] += 1
            return "the answer is 42"

        result = verified_synthesis(
            query="what is the answer?",
            propose_fn=propose,
            oracle=None,
        )
        self.assertEqual(result.total_attempts, 1)
        self.assertEqual(calls["n"], 1)
        self.assertFalse(result.converged)
        self.assertFalse(result.oracle_used)
        self.assertEqual(result.final_answer, "the answer is 42")

    def test_no_oracle_executes_fenced_code(self):
        def propose(prompt, prior):
            return "```python\nprint(2 + 2)\n```"

        result = verified_synthesis(
            query="2+2",
            propose_fn=propose,
            oracle=None,
        )
        self.assertEqual(result.final_answer.strip(), "4")
        self.assertEqual(result.total_attempts, 1)

    # ------------------------------------------------------------------
    # Oracle-driven convergence
    # ------------------------------------------------------------------

    def test_oracle_converges_on_first_pass(self):
        def oracle(q, ans):
            return ("4" in ans, {"checked": ans.strip()})

        def propose(prompt, prior):
            return "```python\nprint(2 + 2)\n```"

        result = verified_synthesis(
            query="2+2",
            propose_fn=propose,
            oracle=oracle,
        )
        self.assertTrue(result.converged)
        self.assertTrue(result.oracle_used)
        self.assertEqual(result.total_attempts, 1)

    def test_oracle_converges_after_self_correction(self):
        # Attempt 1 outputs "5" (wrong); attempt 2 outputs "4" (right).
        seq = ["```python\nprint(5)\n```", "```python\nprint(4)\n```"]

        def propose(prompt, prior):
            i = len(prior)
            return seq[i]

        def oracle(q, ans):
            return ("4" in ans, {"got": ans.strip()})

        result = verified_synthesis(
            query="2+2",
            propose_fn=propose,
            oracle=oracle,
            max_attempts=5,
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.total_attempts, 2)
        # Attempt 2's prompt MUST contain the diagnostic from attempt 1.
        # We can verify indirectly: the second prompt isn't the bare query.
        # (The propose stub above doesn't capture the prompt; a tighter
        # version below covers that.)

    def test_feedback_prompt_includes_oracle_diagnostic(self):
        captured_prompts = []

        def propose(prompt, prior):
            captured_prompts.append(prompt)
            i = len(prior)
            return ["```python\nprint(99)\n```", "```python\nprint(4)\n```"][i]

        def oracle(q, ans):
            return ("4" in ans, {"reason": "answer must contain 4"})

        verified_synthesis(
            query="2+2",
            propose_fn=propose,
            oracle=oracle,
        )
        # First prompt embeds the query and instructs the LLM to emit a
        # fenced python block (Phase 6.1.1 prompt nudge — vanilla CoT
        # cannot use this signal so it is a Δ, not a regression).
        self.assertIn("2+2", captured_prompts[0])
        self.assertIn("```python", captured_prompts[0])
        # Second prompt is feedback and must mention the oracle's reason.
        self.assertIn("answer must contain 4", captured_prompts[1])
        self.assertIn("Previous code", captured_prompts[1])
        self.assertIn("99", captured_prompts[1])

    def test_max_attempts_respected(self):
        calls = {"n": 0}

        def propose(prompt, prior):
            calls["n"] += 1
            return "```python\nprint('nope')\n```"

        def oracle(q, ans):
            return (False, {})

        result = verified_synthesis(
            query="x",
            propose_fn=propose,
            oracle=oracle,
            max_attempts=3,
        )
        self.assertFalse(result.converged)
        self.assertEqual(calls["n"], 3)
        self.assertEqual(result.total_attempts, 3)

    # ------------------------------------------------------------------
    # Robustness
    # ------------------------------------------------------------------

    def test_sandbox_error_does_not_crash_loop(self):
        # Attempt 1 throws; attempt 2 succeeds.
        seq = [
            "```python\nraise RuntimeError('boom')\n```",
            "```python\nprint('ok')\n```",
        ]

        def propose(prompt, prior):
            return seq[len(prior)]

        def oracle(q, ans):
            return ("ok" in ans, {})

        result = verified_synthesis(
            query="x",
            propose_fn=propose,
            oracle=oracle,
            max_attempts=5,
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.total_attempts, 2)
        self.assertIsNotNone(result.attempts[0].error)
        self.assertIsNone(result.attempts[1].error)

    def test_oracle_exception_is_isolated(self):
        # Oracle raises — must be treated as "fail" not "crash".
        def propose(prompt, prior):
            return "```python\nprint('x')\n```"

        def bad_oracle(q, ans):
            raise ValueError("oracle is broken")

        result = verified_synthesis(
            query="x",
            propose_fn=propose,
            oracle=bad_oracle,
            max_attempts=2,
        )
        self.assertFalse(result.converged)
        # Both attempts ran; both have oracle_passed == False with the
        # error captured in oracle_info.
        for a in result.attempts:
            self.assertFalse(a.oracle_passed)
            self.assertIn("oracle_error", a.oracle_info)

    def test_plain_text_response_passes_through(self):
        def propose(prompt, prior):
            return "the answer is 42"

        def oracle(q, ans):
            return ("42" in ans, {})

        result = verified_synthesis(
            query="answer",
            propose_fn=propose,
            oracle=oracle,
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.final_answer, "the answer is 42")
        self.assertEqual(result.attempts[0].extracted_code, "")

    def test_strictly_not_worse_than_vanilla(self):
        """Property test: with oracle disabled, VS == vanilla.

        Vanilla CoT is exactly "make one LLM call and return the
        result".  VS with ``oracle=None`` makes one call and returns
        attempt 1 verbatim — they must be byte-equal.
        """
        VANILLA_RESPONSE = "```python\nprint('hello')\n```"

        def propose(prompt, prior):
            return VANILLA_RESPONSE

        result = verified_synthesis(
            query="say hello",
            propose_fn=propose,
            oracle=None,
        )
        # The VS final answer is the executed output of attempt 1.
        # Vanilla, run through the same sandbox extraction, would
        # produce the same string.  This is the formal "no worse than
        # vanilla" guarantee.
        self.assertEqual(result.total_attempts, 1)
        self.assertEqual(result.final_answer.strip(), "hello")


    # ------------------------------------------------------------------
    # Phase 6.1.1 fixes — surfaced by the user's first live A/B run
    # ------------------------------------------------------------------

    def test_oracle_returning_str_info_does_not_crash(self):
        """Built-in NARE oracles return ``(bool, str)`` — VS must not
        ``dict(str)`` and ValueError out.  Earlier version of synthesis.py
        did exactly that, surfacing as ``ERROR:ValueError`` (or
        ``ERROR:TypeError`` on older Pythons) in the user's A/B routing
        breakdown.
        """
        def propose(prompt, prior):
            return "wrong answer"

        def oracle_str(q, ans):
            # Mirrors string_contains_oracle: (False, "missing substrings: [...]")
            return False, "missing substrings: ['noitingoc']"

        result = verified_synthesis(
            query="reverse 'cognition'",
            propose_fn=propose,
            oracle=oracle_str,
            max_attempts=2,
        )
        self.assertFalse(result.converged)
        # str diagnostic is normalised to a dict so the feedback prompt
        # builder and downstream consumers see a consistent shape.
        self.assertIsInstance(result.attempts[0].oracle_info, dict)
        self.assertIn(
            "noitingoc",
            str(result.attempts[0].oracle_info),
        )

    def test_oracle_returning_none_info_does_not_crash(self):
        """``dict(None)`` raises TypeError — must be normalised, not
        propagated, since a flaky external oracle should never break VS.
        """
        def propose(prompt, prior):
            return "stub"

        def oracle_none(q, ans):
            return False, None

        result = verified_synthesis(
            query="anything",
            propose_fn=propose,
            oracle=oracle_none,
            max_attempts=2,
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.attempts[0].oracle_info, {})

    def test_oracle_returning_int_info_does_not_crash(self):
        """Some oracles encode error codes as ints.  Must not crash."""
        def propose(prompt, prior):
            return "stub"

        def oracle_int(q, ans):
            return False, 42

        result = verified_synthesis(
            query="anything",
            propose_fn=propose,
            oracle=oracle_int,
            max_attempts=1,
        )
        self.assertIsInstance(result.attempts[0].oracle_info, dict)

    def test_all_empty_outputs_falls_back_to_raw_response(self):
        r"""If every attempt's executed_output is empty (LLM kept writing
        code without ``print``), VS must NOT collapse to "" — it must
        fall back to the longest non-empty raw_response so downstream
        consumers (and the user's A/B grader) see something gradeable.

        This is the exact failure mode that produced an empty NARE answer
        in the user's last A/B run on ``code_002``.
        """
        def propose(prompt, prior):
            # Code without ``print`` → safe_execute_freeform returns "".
            return "```python\nx = 'noitigoc'\n```"

        def oracle(q, ans):
            return False, "missing 'noitingoc'"

        result = verified_synthesis(
            query="reverse 'cognition'",
            propose_fn=propose,
            oracle=oracle,
            max_attempts=3,
        )
        self.assertFalse(result.converged)
        # The crucial assertion: final_answer must NOT be empty/whitespace.
        self.assertTrue(result.final_answer.strip())
        self.assertIn("noitigoc", result.final_answer)


    # ------------------------------------------------------------------
    # Phase 6.1.3 — bound NARE >= attempt 1 (no regression vs vanilla)
    # ------------------------------------------------------------------

    def test_no_convergence_returns_attempt_1_payload_not_attempt_n(self):
        """Hardening of the 'never worse than vanilla' contract.

        On the user's hard_007 (sum of digits of 30!), VS attempt 1
        produced ``factorial(30)`` (a stable wrong answer that vanilla
        would have returned and stopped at).  Earlier VS retries with
        rising temperature kept producing similarly wrong outputs, and
        the old ``max(..., key=executed_output length)`` tiebreaker
        sometimes picked attempt 5's noise over attempt 1.

        New contract: when no attempt converges and attempt 1 produced
        *some* executable output, return attempt 1's output.  This
        guarantees NARE >= the vanilla single-shot floor, with retries
        as upside-only.
        """
        attempt1_out = "265252859812191058636308480000000"  # factorial(30)
        attempt5_out = "0"  # noise

        seq = [
            f"```python\nimport math\nprint(math.factorial(30))\n```",  # attempt 1 → 30!
            f"```python\nprint(0)\n```",  # attempt 2 → 0
            f"```python\nprint(0)\n```",  # attempt 3 → 0
            f"```python\nprint(0)\n```",  # attempt 4 → 0
            f"```python\nprint(0)\n```",  # attempt 5 → 0
        ]

        def propose(prompt, prior):
            return seq[len(prior)]

        def oracle(q, ans):
            return False, "missing expected numbers: [117]"

        result = verified_synthesis(
            query="sum of digits of 30!",
            propose_fn=propose,
            oracle=oracle,
            max_attempts=5,
        )
        self.assertFalse(result.converged)
        # MUST be attempt 1's output, not attempt 5's noise.
        self.assertEqual(result.final_answer, attempt1_out)
        self.assertNotEqual(result.final_answer, attempt5_out)

    def test_no_convergence_uses_retry_when_attempt1_empty(self):
        """If attempt 1 produced no executable output (LLM forgot
        ``print``), but a later attempt did, switch to that retry.
        This is the upside path — VS legitimately fixed a defect.
        """
        seq = [
            "```python\nx = 5\n```",  # no print → empty
            "```python\nprint(34650)\n```",  # has print → "34650"
            "```python\nprint(34650)\n```",
        ]

        def propose(prompt, prior):
            return seq[min(len(prior), len(seq) - 1)]

        def oracle(q, ans):
            # Always rejects so we go past convergence into fallback.
            return False, "wrong"

        result = verified_synthesis(
            query="permutations MISSISSIPPI",
            propose_fn=propose,
            oracle=oracle,
            max_attempts=3,
        )
        self.assertFalse(result.converged)
        # Floor is empty → switched to retry that produced output.
        self.assertEqual(result.final_answer, "34650")

    def test_no_convergence_floor_holds_when_retries_drift(self):
        """Counterexample to the 'always pick longest' tiebreaker.

        Attempt 1 prints '117' (correct semantics, oracle still rejects
        because of an outer wrap).  Attempts 2-5 print noise.  We must
        return '117' — the stable attempt-1 output — not a longer
        garbled retry.
        """
        seq = [
            "```python\nprint(117)\n```",                # attempt 1 → 117
            "```python\nprint('not the answer at all')\n```",  # longer noise
            "```python\nprint('not the answer at all either')\n```",
            "```python\nprint('and another wrong one')\n```",
            "```python\nprint('and a fifth')\n```",
        ]

        def propose(prompt, prior):
            return seq[len(prior)]

        def oracle(q, ans):
            return False, "rejected"

        result = verified_synthesis(
            query="something",
            propose_fn=propose,
            oracle=oracle,
            max_attempts=5,
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.final_answer, "117")

    def test_empty_output_feedback_mentions_print(self):
        """When attempt 1's code ran without errors but produced no
        stdout (no print), the feedback prompt to attempt 2 must
        explicitly tell the LLM to add ``print(...)``.  Without this
        nudge Gemma frequently regenerates near-identical code and
        the loop stalls (the user's hard_010 failure mode).
        """
        captured_prompts: List[str] = []

        def propose(prompt, prior):
            captured_prompts.append(prompt)
            i = len(prior)
            if i == 0:
                # No print() — sandbox captures empty output.
                return "```python\nx = 42\n```"
            # Attempt 2: pass.
            return "```python\nprint(42)\n```"

        def oracle(q, ans):
            return ans.strip() == "42", "ok" if ans.strip() == "42" else "missing"

        result = verified_synthesis(
            query="compute 42",
            propose_fn=propose,
            oracle=oracle,
            max_attempts=3,
        )
        self.assertTrue(result.converged)
        # Attempt 2's prompt must include the print() nudge.
        self.assertTrue(len(captured_prompts) >= 2)
        feedback = captured_prompts[1]
        self.assertIn("print", feedback)
        self.assertIn("did not call", feedback)


if __name__ == "__main__":
    unittest.main()
