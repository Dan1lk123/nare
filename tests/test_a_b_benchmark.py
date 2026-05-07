"""Unit tests for the LLM-free parts of benchmarks/a_b_benchmark.py.

The script's network paths (NARE solve, vanilla Gemini call) are
covered by the existing integration tests; these tests focus on the
statistics helpers, the report renderer, and end-to-end wiring with a
stubbed agent + stubbed vanilla runner so CI runs offline."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

ab = importlib.import_module("benchmarks.a_b_benchmark")


def test_wilson_ci_basic_bounds():
    low, high = ab.wilson_ci(0, 0)
    assert (low, high) == (0.0, 0.0)

    low, high = ab.wilson_ci(5, 10)
    assert 0.0 <= low <= 0.5 <= high <= 1.0

    low, high = ab.wilson_ci(10, 10)
    # All-correct n=10 must give a CI that strictly does not include 0
    # but also need not include exactly 1.0 (Wilson is conservative).
    assert low > 0.5
    assert high == pytest.approx(1.0, abs=1e-9)


def test_paired_bootstrap_diff_ci_basic():
    # NARE wins on every task → difference must be strictly positive.
    pairs = [(True, False)] * 20
    point, low, high = ab.paired_bootstrap_diff_ci(pairs, n_resamples=2000)
    assert point == pytest.approx(1.0)
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)


def test_paired_bootstrap_diff_ci_no_difference():
    pairs = [(True, True)] * 5 + [(False, False)] * 5
    point, low, high = ab.paired_bootstrap_diff_ci(pairs, n_resamples=2000)
    assert point == pytest.approx(0.0)
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(0.0)


def test_paired_bootstrap_handles_empty():
    point, low, high = ab.paired_bootstrap_diff_ci([], n_resamples=100)
    assert (point, low, high) == (0.0, 0.0, 0.0)


def test_grade_with_numeric_set_oracle():
    spec = {"type": "numeric_set", "expected": [42]}
    ok, _info = ab._grade("the answer is 42", spec, "what?")
    assert ok is True

    ok, _info = ab._grade("definitely 17", spec, "what?")
    assert ok is False


def test_grade_with_bad_oracle_spec_returns_false():
    spec = {"type": "definitely_not_an_oracle"}
    ok, info = ab._grade("anything", spec, "q")
    assert ok is False
    assert "bad oracle_spec" in info


def test_load_dataset_validates_required_keys(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"id": "1", "query": "q"}]))  # no oracle_spec
    with pytest.raises(ValueError, match="oracle_spec"):
        ab._load_dataset(str(bad))


def test_load_dataset_returns_builtin_when_path_missing():
    ds = ab._load_dataset(None)
    assert len(ds) == len(ab.BUILTIN_TASKS)
    # All built-in tasks must be valid (every benchmark depends on this).
    for t in ds:
        assert "id" in t
        assert "query" in t
        assert "oracle_spec" in t
        # build_oracle_from_spec must accept the built-in spec.
        from nare.oracle import build_oracle_from_spec
        build_oracle_from_spec(t["oracle_spec"])


def test_build_report_end_to_end(tmp_path):
    """Produce a fake result list and confirm the report writer emits
    JSON + Markdown with the right summary fields."""
    results = [
        ab.TaskResult(
            task_id=f"t{i}",
            category=("GSM8K" if i < 3 else "Logic"),
            seed=0,
            paraphrase_idx=0,
            base_task_id=f"t{i}",
            query=f"query{i}",
            nare_route=("FAST" if i == 0 else "SLOW"),
            nare_answer=f"answer{i}",
            nare_correct=(i != 4),  # 4/5 correct
            nare_latency_s=1.0 + 0.1 * i,
            nare_oracle_info="ok" if i != 4 else "wrong",
            vanilla_answer=f"vanilla{i}",
            vanilla_correct=(i < 2),  # 2/5 correct
            vanilla_latency_s=0.5,
            vanilla_oracle_info="ok",
            expected_solution=str(i),
        )
        for i in range(5)
    ]

    json_path, md_path = ab.build_report(
        results,
        output_dir=str(tmp_path),
        n_seeds=1,
        n_tasks=5,
    )

    with open(json_path, encoding="utf-8") as f:
        report = json.load(f)

    assert report["n_total"] == 5
    assert report["nare"]["accuracy"] == pytest.approx(0.8)
    assert report["vanilla"]["accuracy"] == pytest.approx(0.4)
    # Diff point estimate must equal 0.8 - 0.4 = 0.4.
    assert report["diff"]["nare_minus_vanilla"] == pytest.approx(0.4)
    # Amortized = the one FAST task / 5 = 20%.
    assert report["nare"]["amortized_pct"] == pytest.approx(20.0)
    # Per-category present.
    assert set(report["per_category"].keys()) == {"GSM8K", "Logic"}

    md = open(md_path, encoding="utf-8").read()
    assert "# A/B Benchmark" in md
    assert "Δ (NARE − Vanilla)" in md
    assert "GSM8K" in md
    assert "FAST" in md  # routing breakdown


def test_run_a_b_with_stubbed_arms(monkeypatch, tmp_path):
    """Drive run_a_b end-to-end without any real LLM/agent calls."""
    # Stub vanilla solve.
    monkeypatch.setattr(
        ab,
        "vanilla_gemini_solve",
        lambda q: ("vanilla says 42", 0.42),
    )

    # Stub the agent: NAREProductionAgent constructed inside
    # _build_isolated_agent. We patch _build_isolated_agent to return a
    # tiny duck-typed agent.
    class _StubAgent:
        def solve(self, query, oracle_spec=None):
            return {
                "final_answer": "nare says 42",
                "route_decision": "SLOW",
            }

    monkeypatch.setattr(
        ab, "_build_isolated_agent", lambda persist_dir: _StubAgent()
    )

    tasks = [
        {
            "id": "x1",
            "category": "GSM8K",
            "query": "what is the answer?",
            "expected_solution": "42",
            "oracle_spec": {"type": "string_contains", "must_contain": ["42"]},
        },
        {
            "id": "x2",
            "category": "GSM8K",
            "query": "another question?",
            "expected_solution": "42",
            "oracle_spec": {"type": "string_contains", "must_contain": ["42"]},
        },
    ]

    results = ab.run_a_b(
        tasks=tasks,
        seeds=[0, 1],
        persist_dir=None,
    )
    # 2 tasks × 2 seeds = 4 results.
    assert len(results) == 4
    # Both arms should grade as correct (both answers contain "42").
    assert all(r.nare_correct for r in results)
    assert all(r.vanilla_correct for r in results)
    assert all(r.nare_route == "SLOW" for r in results)

    # Report writes successfully and JSON parses.
    json_path, md_path = ab.build_report(
        results,
        output_dir=str(tmp_path),
        n_seeds=2,
        n_tasks=2,
    )
    report = json.loads(open(json_path, encoding="utf-8").read())
    assert report["nare"]["accuracy"] == 1.0
    assert report["vanilla"]["accuracy"] == 1.0


# ---------------------------------------------------------------------
# Paraphrase mode (PR #18)
# ---------------------------------------------------------------------


def test_expand_paraphrases_p1_returns_originals_only():
    tasks = [
        {"id": "a", "query": "Q-A", "paraphrases": ["alt-A1", "alt-A2"], "oracle_spec": {}},
        {"id": "b", "query": "Q-B", "oracle_spec": {}},
    ]
    out = ab._expand_paraphrases(tasks, paraphrases=1)
    assert [(t["task_id"], t["query"], t["paraphrase_idx"]) for t in out] == [
        ("a#p0", "Q-A", 0),
        ("b#p0", "Q-B", 0),
    ]
    # All emitted instances carry their base task id.
    assert all(t["base_task_id"] in {"a", "b"} for t in out)


def test_expand_paraphrases_p3_emits_consecutive_paraphrases():
    tasks = [
        {"id": "a", "query": "Q-A", "paraphrases": ["alt-A1", "alt-A2"], "oracle_spec": {}},
        {"id": "b", "query": "Q-B", "paraphrases": ["alt-B1", "alt-B2"], "oracle_spec": {}},
    ]
    out = ab._expand_paraphrases(tasks, paraphrases=3)
    # Crucial property: paraphrases of the same base task are emitted
    # consecutively so the NARE memory store warms between them.
    assert [t["task_id"] for t in out] == [
        "a#p0", "a#p1", "a#p2",
        "b#p0", "b#p1", "b#p2",
    ]
    assert [t["query"] for t in out] == [
        "Q-A", "alt-A1", "alt-A2",
        "Q-B", "alt-B1", "alt-B2",
    ]


def test_expand_paraphrases_caps_at_available_count():
    """If a task has fewer paraphrases than requested, we silently cap
    instead of erroring — so a heterogeneous dataset still runs."""
    tasks = [
        {"id": "a", "query": "Q-A", "paraphrases": ["alt-A1"], "oracle_spec": {}},
        {"id": "b", "query": "Q-B", "oracle_spec": {}},  # no paraphrases
    ]
    out = ab._expand_paraphrases(tasks, paraphrases=5)
    # Task a has 1 original + 1 paraphrase = 2 variants, capped at 5.
    # Task b has just the original.
    assert [t["task_id"] for t in out] == ["a#p0", "a#p1", "b#p0"]


def test_run_a_b_with_paraphrases_keeps_memory_warm(monkeypatch, tmp_path):
    """End-to-end: when paraphrases > 1, the same agent instance must
    receive all the paraphrased queries of one base task in order
    (memory warming is the whole point of the mode)."""
    seen_queries: list = []

    class _StubAgent:
        def solve(self, query, oracle_spec=None):
            seen_queries.append(query)
            return {"final_answer": "42", "route_decision": "SLOW"}

    # One stub agent per seed — when run_a_b reuses the agent across
    # paraphrases we'll see all queries on the same agent instance.
    stub_instances: list = []

    def _make_stub(persist_dir):
        a = _StubAgent()
        stub_instances.append(a)
        return a

    monkeypatch.setattr(ab, "_build_isolated_agent", _make_stub)
    monkeypatch.setattr(ab, "vanilla_gemini_solve", lambda q: ("42", 0.1))

    tasks = [
        {
            "id": "a",
            "category": "GSM8K",
            "query": "Q-A",
            "paraphrases": ["alt-A1", "alt-A2"],
            "expected_solution": "42",
            "oracle_spec": {"type": "string_contains", "must_contain": ["42"]},
        },
        {
            "id": "b",
            "category": "GSM8K",
            "query": "Q-B",
            "paraphrases": ["alt-B1"],
            "expected_solution": "42",
            "oracle_spec": {"type": "string_contains", "must_contain": ["42"]},
        },
    ]

    results = ab.run_a_b(
        tasks=tasks, seeds=[0], persist_dir=None, paraphrases=3,
    )

    # 3 paraphrases on task a + 2 on task b (capped) = 5 results.
    assert len(results) == 5
    # Exactly one stub agent created for the single seed → memory was
    # actually shared across paraphrases.
    assert len(stub_instances) == 1
    # The stub agent saw all 5 queries in the canonical order.
    assert seen_queries == ["Q-A", "alt-A1", "alt-A2", "Q-B", "alt-B1"]
    # Paraphrase indices and base task ids are recorded correctly.
    assert [(r.base_task_id, r.paraphrase_idx) for r in results] == [
        ("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1),
    ]


def test_build_report_emits_learning_curve_when_paraphrases_present(tmp_path):
    """The Learning curve / Per-task trajectory sections are gated on
    presence of paraphrase_idx > 0 — i.e. only render in paraphrase
    mode, never in legacy single-query mode."""
    results = [
        ab.TaskResult(
            task_id="a#p0", category="GSM8K", seed=0,
            paraphrase_idx=0, base_task_id="a", query="Q-A",
            nare_route="SLOW", nare_answer="42", nare_correct=True,
            nare_latency_s=6.0, nare_oracle_info="ok",
            vanilla_answer="42", vanilla_correct=True,
            vanilla_latency_s=2.0, vanilla_oracle_info="ok",
            expected_solution="42",
        ),
        ab.TaskResult(
            task_id="a#p1", category="GSM8K", seed=0,
            paraphrase_idx=1, base_task_id="a", query="alt-A1",
            nare_route="FAST", nare_answer="42", nare_correct=True,
            nare_latency_s=0.7, nare_oracle_info="ok",
            vanilla_answer="42", vanilla_correct=True,
            vanilla_latency_s=2.1, vanilla_oracle_info="ok",
            expected_solution="42",
        ),
    ]
    json_path, md_path = ab.build_report(
        results, output_dir=str(tmp_path), n_seeds=1, n_tasks=1,
    )
    md = open(md_path, encoding="utf-8").read()
    assert "Learning curve" in md
    assert "Per-task trajectory" in md
    # The trajectory string must show progression "SLOW → FAST".
    assert "SLOW(6.0s) → FAST(0.7s)" in md


def test_build_report_omits_learning_curve_when_no_paraphrases(tmp_path):
    """In legacy mode (paraphrase_idx all zero) the report must not
    emit the new sections — keeps the existing single-query report
    layout untouched."""
    results = [
        ab.TaskResult(
            task_id="a", category="GSM8K", seed=0,
            paraphrase_idx=0, base_task_id="a", query="Q-A",
            nare_route="SLOW", nare_answer="42", nare_correct=True,
            nare_latency_s=6.0, nare_oracle_info="ok",
            vanilla_answer="42", vanilla_correct=True,
            vanilla_latency_s=2.0, vanilla_oracle_info="ok",
            expected_solution="42",
        ),
    ]
    json_path, md_path = ab.build_report(
        results, output_dir=str(tmp_path), n_seeds=1, n_tasks=1,
    )
    md = open(md_path, encoding="utf-8").read()
    assert "Learning curve" not in md
    assert "Per-task trajectory" not in md


def test_per_task_trajectory_separates_seeds():
    """When the same base task is graded under multiple seeds, the
    per-task trajectory section must render one row per (base, seed)
    pair — not one merged row interleaving results from different
    independent NARE agents."""
    import tempfile

    def _r(base, p, seed, route, lat):
        return ab.TaskResult(
            task_id=f"{base}#p{p}", category="GSM8K", seed=seed,
            paraphrase_idx=p, base_task_id=base, query=f"q-{p}-{seed}",
            nare_route=route, nare_answer="42", nare_correct=True,
            nare_latency_s=lat, nare_oracle_info="ok",
            vanilla_answer="42", vanilla_correct=True,
            vanilla_latency_s=2.0, vanilla_oracle_info="ok",
            expected_solution="42",
        )

    results = [
        # seed=0: SLOW then FAST
        _r("a", 0, 0, "SLOW", 6.0),
        _r("a", 1, 0, "FAST", 0.7),
        # seed=1: SLOW then SLOW (independent agent — never warmed)
        _r("a", 0, 1, "SLOW", 6.1),
        _r("a", 1, 1, "SLOW", 5.9),
    ]

    with tempfile.TemporaryDirectory() as td:
        _, md_path = ab.build_report(
            results, output_dir=td, n_seeds=2, n_tasks=2,
        )
        md = open(md_path, encoding="utf-8").read()

    # Each (base_task_id, seed) must produce its own trajectory row,
    # NOT one merged row blending the four results.
    assert "SLOW(6.0s) → FAST(0.7s)" in md, (
        "seed=0 trajectory missing; seeds were probably merged."
    )
    assert "SLOW(6.1s) → SLOW(5.9s)" in md, (
        "seed=1 trajectory missing; seeds were probably merged."
    )
    # Header should include a Seed column when n_seeds > 1.
    assert "| Base task | Seed | Category | Trajectory |" in md


def test_per_task_trajectory_omits_seed_column_for_single_seed():
    """When only one seed is present, the trajectory table should keep
    the original 3-column layout for backwards compatibility with
    single-seed report consumers."""
    import tempfile

    results = [
        ab.TaskResult(
            task_id="a#p0", category="GSM8K", seed=0,
            paraphrase_idx=0, base_task_id="a", query="q",
            nare_route="SLOW", nare_answer="42", nare_correct=True,
            nare_latency_s=6.0, nare_oracle_info="ok",
            vanilla_answer="42", vanilla_correct=True,
            vanilla_latency_s=2.0, vanilla_oracle_info="ok",
            expected_solution="42",
        ),
        ab.TaskResult(
            task_id="a#p1", category="GSM8K", seed=0,
            paraphrase_idx=1, base_task_id="a", query="q",
            nare_route="FAST", nare_answer="42", nare_correct=True,
            nare_latency_s=0.7, nare_oracle_info="ok",
            vanilla_answer="42", vanilla_correct=True,
            vanilla_latency_s=2.0, vanilla_oracle_info="ok",
            expected_solution="42",
        ),
    ]

    with tempfile.TemporaryDirectory() as td:
        _, md_path = ab.build_report(
            results, output_dir=td, n_seeds=1, n_tasks=1,
        )
        md = open(md_path, encoding="utf-8").read()

    # No seed column in single-seed mode.
    assert "| Base task | Category | Trajectory |" in md
    assert "| Base task | Seed | Category | Trajectory |" not in md


def test_builtin_tasks_have_paraphrases():
    """Smoke-test: every shipped built-in task carries at least one
    paraphrase, otherwise --paraphrases >1 silently degrades."""
    for task in ab.BUILTIN_TASKS:
        ps = task.get("paraphrases", []) or []
        assert len(ps) >= 2, (
            f"task {task['id']} should ship at least 2 paraphrases for "
            f"--paraphrases 3 mode to be fully populated; got {len(ps)}."
        )
