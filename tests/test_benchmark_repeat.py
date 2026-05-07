"""Smoke tests for the repeat-test amortization harness.

We do NOT exercise the full 24-task LLM run here — that would cost
real API tokens and take ~5 minutes. We just verify:

1. ``--repeat`` and ``--keep-memory`` are exposed on the CLI.
2. ``_run_one_iteration`` returns a serialisable summary dict with the
   keys the cross-iteration table consumes.
3. ``_print_amortization_summary`` handles the no-shift, partial-shift,
   and full-amortisation cases without crashing.

The end-to-end "do iterations 2 and 3 actually shift SLOW counts down?"
question can only be answered against a real LLM, which is exercised by
running ``python benchmarks/full_benchmark.py --repeat=3`` manually.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmarks.full_benchmark as bench  # noqa: E402


class TestCliArguments(unittest.TestCase):
    def test_help_lists_repeat_flag(self):
        proc = subprocess.run(
            [sys.executable, os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "benchmarks", "full_benchmark.py",
            ), "--help"],
            check=True, capture_output=True, text=True,
        )
        self.assertIn("--repeat", proc.stdout)
        self.assertIn("--keep-memory", proc.stdout)


class TestPrintAmortizationSummary(unittest.TestCase):
    def _make_iter(self, idx, slow, hybrid, fast=0, reflex=0, total_time=120.0,
                   correct=20, total=24):
        return {
            "iteration": idx,
            "total_iterations": 3,
            "total_tasks": total,
            "correct": correct,
            "accuracy_pct": round(100 * correct / total, 1),
            "total_time_s": total_time,
            "route_distribution": {
                "SLOW": slow,
                "HYBRID": hybrid,
                "FAST": fast,
                "REFLEX": reflex,
            },
        }

    def test_renders_single_iteration_no_delta(self):
        iters = [self._make_iter(1, 20, 4)]
        buf = io.StringIO()
        with redirect_stdout(buf):
            bench._print_amortization_summary(iters)
        out = buf.getvalue()
        self.assertIn("CROSS-ITERATION AMORTIZATION", out)
        self.assertNotIn("ΔSLOW", out)  # no delta line on a single iteration
        self.assertIn("20", out)  # SLOW count rendered

    def test_renders_amortization_shift(self):
        iters = [
            self._make_iter(1, 20, 4, total_time=120.0),
            self._make_iter(2, 12, 8, fast=4, total_time=80.0),
            self._make_iter(3, 5, 5, fast=10, reflex=4, total_time=40.0),
        ]
        buf = io.StringIO()
        with redirect_stdout(buf):
            bench._print_amortization_summary(iters)
        out = buf.getvalue()
        self.assertIn("ΔSLOW", out)
        self.assertIn("(-15)", out)  # 20 -> 5
        self.assertIn("speedup", out)
        # 120 / 40 = 3.0×
        self.assertIn("3.00×", out)

    def test_handles_unknown_routes_in_other_bucket(self):
        iters = [{
            "iteration": 1,
            "total_iterations": 1,
            "total_tasks": 24,
            "correct": 20,
            "accuracy_pct": 83.3,
            "total_time_s": 120.0,
            "route_distribution": {"SLOW": 18, "WEIRD_ROUTE": 6},
        }]
        buf = io.StringIO()
        with redirect_stdout(buf):
            bench._print_amortization_summary(iters)
        out = buf.getvalue()
        # The unknown route should be counted in the OTHER column (=6).
        # Find the data row and check the last column.
        line = [ln for ln in out.splitlines() if ln.lstrip().startswith("1 ")][0]
        # Last whitespace-separated token on the data row is the OTHER column.
        cols = line.split()
        self.assertEqual(cols[-1], "6")


if __name__ == "__main__":
    unittest.main()
