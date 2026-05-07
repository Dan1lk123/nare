"""Regression tests for the REM-repair validator-aware path.

The benchmark log showed the classic "stuck repair" symptom:

    [REM Repair] Attempt 1: generated repaired code (914 chars)
    [REM] 'Email Extraction' repair did not improve (old=0.10, new=0.10)

repeated 3× with identical scores. Root cause: ``repair_skill(max_attempts=2)``
returned on the **first** successfully-extracted candidate regardless of
validation outcome, so the caller's max_attempts loop was effectively dead
code.

These tests assert:

1. Without ``validator``: legacy behaviour preserved — return first repair.
2. With ``validator``: iterate up to ``max_attempts``, return the best.
3. With ``validator``: if no candidate beats baseline, return original code
   (so the caller can correctly distinguish "no improvement" from
   "I returned a repair that happens to be the same score").
4. Early-stop when a candidate scores ≥ 0.80.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from nare import llm


def _make_response(code: str) -> dict:
    """Build the minimal Gemini response shape consumed by repair_skill."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": f"```python\n{code}\n```"}
                    ]
                }
            }
        ]
    }


class TestRepairSkillValidatorPath(unittest.TestCase):
    SAMPLE_CODE = "def execute(q):\n    return q.upper()\n"

    def setUp(self):
        # Bypass API-key check.
        self._api_patch = patch.object(llm, "_ensure_api_key", lambda: None)
        self._api_patch.start()

    def tearDown(self):
        self._api_patch.stop()

    # --- 1. Legacy path: no validator → return first repair --------------
    def test_legacy_no_validator_returns_first_repair(self):
        responses = [
            _make_response("def execute(q): return 'A'"),
            _make_response("def execute(q): return 'B'"),
        ]
        with patch.object(llm, "_post", side_effect=responses) as mock_post:
            result = llm.repair_skill(
                self.SAMPLE_CODE, "TestSkill", [],
                "err", {"overall": 0.1}, max_attempts=3,
            )
        self.assertIn("return 'A'", result)
        # Stops at first successful generation.
        self.assertEqual(mock_post.call_count, 1)

    # --- 2. Validator: iterate and return the best ----------------------
    def test_validator_iterates_and_returns_best(self):
        responses = [
            _make_response("def execute(q): return 'WORST'"),
            _make_response("def execute(q): return 'BEST'"),
            _make_response("def execute(q): return 'MID'"),
        ]
        scores = {
            "WORST": 0.05,
            "BEST": 0.65,
            "MID": 0.40,
        }

        def _validator(code: str) -> float:
            for marker, sc in scores.items():
                if marker in code:
                    return sc
            return 0.0

        with patch.object(llm, "_post", side_effect=responses) as mock_post:
            result = llm.repair_skill(
                self.SAMPLE_CODE, "TestSkill", [],
                "err", {"overall": 0.10}, max_attempts=3,
                validator=_validator, baseline_score=0.10,
            )
        self.assertIn("'BEST'", result)
        # All three attempts should have run (no early-stop hit since 0.65 < 0.80).
        self.assertEqual(mock_post.call_count, 3)

    # --- 3. Validator: no candidate beats baseline → return original ----
    def test_validator_no_improvement_returns_original(self):
        responses = [
            _make_response("def execute(q): return 'NOPE1'"),
            _make_response("def execute(q): return 'NOPE2'"),
        ]

        def _validator(code: str) -> float:
            return 0.05  # always worse than baseline 0.30

        with patch.object(llm, "_post", side_effect=responses):
            result = llm.repair_skill(
                self.SAMPLE_CODE, "TestSkill", [],
                "err", {"overall": 0.30}, max_attempts=2,
                validator=_validator, baseline_score=0.30,
            )
        self.assertEqual(result, self.SAMPLE_CODE)

    # --- 4. Validator: early-stop on score >= 0.80 ----------------------
    def test_validator_early_stops_on_high_score(self):
        responses = [
            _make_response("def execute(q): return 'HIGH'"),
            _make_response("def execute(q): return 'NEVER_REACHED'"),
        ]

        def _validator(code: str) -> float:
            return 0.95 if "HIGH" in code else 0.10

        with patch.object(llm, "_post", side_effect=responses) as mock_post:
            result = llm.repair_skill(
                self.SAMPLE_CODE, "TestSkill", [],
                "err", {"overall": 0.10}, max_attempts=3,
                validator=_validator, baseline_score=0.10,
            )
        self.assertIn("'HIGH'", result)
        self.assertEqual(mock_post.call_count, 1)

    # --- 5. Validator: handles raising-validator gracefully -------------
    def test_validator_exception_treated_as_low_score(self):
        responses = [
            _make_response("def execute(q): return 'CRASHES'"),
            _make_response("def execute(q): return 'OK'"),
        ]

        def _validator(code: str) -> float:
            if "CRASHES" in code:
                raise ValueError("simulated validator crash")
            return 0.50

        with patch.object(llm, "_post", side_effect=responses):
            result = llm.repair_skill(
                self.SAMPLE_CODE, "TestSkill", [],
                "err", {"overall": 0.10}, max_attempts=2,
                validator=_validator, baseline_score=0.10,
            )
        self.assertIn("'OK'", result)


if __name__ == "__main__":
    unittest.main()
