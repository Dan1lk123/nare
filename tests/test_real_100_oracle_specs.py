"""Offline tests for the oracle_spec construction in the
benchmarks/real_100_benchmark.py harness.

We only test the pure-logic helpers (no LLM, no agent). The helpers
must produce specs that:
  1. ``build_oracle_from_spec`` accepts (no exceptions);
  2. correctly grade obvious right / wrong answers."""

from __future__ import annotations

import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from nare.oracle import build_oracle_from_spec

bench = importlib.import_module("benchmarks.real_100_benchmark")


# ---------- GSM8K ----------

def test_gsm8k_numeric_spec_accepts_correct_answer():
    task = {"q": "Maria has 3 + 4 apples?", "a": "7", "domain": "GSM8K"}
    spec = bench.build_oracle_spec(task)
    assert spec == {"type": "numeric_set", "expected": [7.0]}
    oracle = build_oracle_from_spec(spec)
    assert oracle("q", "the answer is 7")[0] is True
    assert oracle("q", "the answer is 8")[0] is False


def test_gsm8k_numeric_spec_handles_commas():
    task = {"q": "?", "a": "1,234", "domain": "GSM8K"}
    spec = bench.build_oracle_spec(task)
    assert spec == {"type": "numeric_set", "expected": [1234.0]}


def test_gsm8k_returns_none_when_answer_not_numeric():
    task = {"q": "?", "a": "low-quality data", "domain": "GSM8K"}
    assert bench.build_oracle_spec(task) is None


# ---------- HumanEval-Lite ----------

def test_humaneval_prime_harness_accepts_real_prime_function():
    task = {
        "q": "Write a Python function to check if a number is prime.",
        "a": "prime",
        "domain": "HumanEval-Lite",
    }
    spec = bench.build_oracle_spec(task)
    assert spec["type"] == "python_assert"
    oracle = build_oracle_from_spec(spec)

    good = (
        "Here is a prime check:\n"
        "def is_prime(n):\n"
        "    if n < 2: return False\n"
        "    for i in range(2, n):\n"
        "        if n % i == 0:\n"
        "            return False\n"
        "    return True\n"
    )
    ok, info = oracle("q", good)
    assert ok, info


def test_humaneval_prime_harness_rejects_wrong_function():
    task = {
        "q": "Write a Python function to check if a number is prime.",
        "a": "prime",
        "domain": "HumanEval-Lite",
    }
    oracle = build_oracle_from_spec(bench.build_oracle_spec(task))
    ok, _ = oracle("q", "def add(a, b): return a + b")
    assert ok is False


def test_humaneval_reverse_string_harness_rejects_slicing_shortcut():
    """The task explicitly forbids ``[::-1]`` — the harness must catch it."""
    task = {
        "q": "Write a Python function to reverse a string without using slicing [::-1].",
        "a": "reverse",
        "domain": "HumanEval-Lite",
    }
    oracle = build_oracle_from_spec(bench.build_oracle_spec(task))

    bad = "def reverse_string(s):\n    return s[::-1]\n"
    ok, info = oracle("q", bad)
    assert ok is False
    assert "slicing" in info or "[::-1]" in info

    good = (
        "def reverse_string(s):\n"
        "    out = ''\n"
        "    for ch in s:\n"
        "        out = ch + out\n"
        "    return out\n"
    )
    ok2, info2 = oracle("q", good)
    assert ok2, info2


def test_humaneval_falls_back_to_logic_spec_when_no_harness():
    """Unknown coding task → fall back to any-of string match."""
    task = {
        "q": "Write a Python function to do something exotic.",
        "a": "exotic",
        "domain": "HumanEval-Lite",
    }
    spec = bench.build_oracle_spec(task)
    assert spec is not None
    # Must be a python_assert any-of spec (since "exotic" is non-numeric).
    assert spec["type"] == "python_assert"
    oracle = build_oracle_from_spec(spec)
    assert oracle("q", "something exotic was here")[0] is True
    assert oracle("q", "totally unrelated")[0] is False


def test_humaneval_variation_suffix_still_routes_to_correct_harness():
    """Ensure ``Write a function to count vowels (Variation 2)`` still
    maps to the vowels harness, not the fallback."""
    task = {
        "q": "Write a Python function to count the number of vowels in a string. (Variation 2)",
        "a": "vowels/count",
        "domain": "HumanEval-Lite",
    }
    spec = bench.build_oracle_spec(task)
    assert spec["type"] == "python_assert"
    # The fallback for "vowels/count" would be ANY-of substring; the
    # vowels harness explicitly checks for `def`. A bare keyword-only
    # answer would pass the fallback but fail the harness:
    oracle = build_oracle_from_spec(spec)
    bare = "Just the word vowels and count."
    assert oracle("q", bare)[0] is False  # harness rejects, fallback would pass


# ---------- Logic/QA ----------

def test_logic_numeric_only_alternatives_use_numeric_set():
    spec = bench.build_oracle_spec(
        {"q": "?", "a": "32", "domain": "Logic/QA"}
    )
    assert spec == {"type": "numeric_set", "expected": [32.0]}


def test_logic_mixed_alternatives_become_python_assert_anyof():
    spec = bench.build_oracle_spec(
        {"q": "?", "a": "12/all", "domain": "Logic/QA"}
    )
    assert spec["type"] == "python_assert"
    oracle = build_oracle_from_spec(spec)
    # Either alternative must satisfy.
    assert oracle("q", "the answer is 12 months")[0] is True
    assert oracle("q", "all of them, of course")[0] is True
    assert oracle("q", "definitely 7 months")[0] is False


def test_logic_percentage_parsed_into_numeric():
    spec = bench.build_oracle_spec(
        {"q": "?", "a": "0.25/1/4/25%", "domain": "Logic/QA"}
    )
    assert spec["type"] == "numeric_set"
    nums = sorted(spec["expected"])
    # Both 0.25 and 25% (==0.25) collapse, and 1, 4 stay distinct.
    assert 0.25 in nums
    assert 1.0 in nums
    assert 4.0 in nums


def test_logic_textual_singleton():
    spec = bench.build_oracle_spec(
        {"q": "Capital of France?", "a": "Paris", "domain": "Logic/QA"}
    )
    oracle = build_oracle_from_spec(spec)
    assert oracle("q", "the capital is paris")[0] is True
    assert oracle("q", "the capital is london")[0] is False


# ---------- Unknown domain ----------

def test_unknown_domain_returns_none():
    assert bench.build_oracle_spec({"q": "?", "a": "x", "domain": "???"}) is None
