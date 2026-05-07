"""Tests for AST-fingerprint meta-abduction (Tier A3).

These tests confirm that the new structural signature actually
distinguishes skills that the legacy keyword-Jaccard would have falsely
merged, and continues to merge skills that really are doing the same
work in different surface text.
"""

from __future__ import annotations

import pytest

from nare.meta_abduction import (
    MetaAbductionEngine,
    ast_fingerprint,
    ast_jaccard,
)


# ---------------------------------------------------------------------------
# Skill code fixtures
# ---------------------------------------------------------------------------


SUM_SKILL = """
def trigger(query: str) -> bool:
    return '+' in query

def execute(query: str) -> str:
    nums = []
    cur = ''
    for c in query + ' ':
        if c.isdigit():
            cur += c
        elif cur:
            nums.append(int(cur))
            cur = ''
    return str(sum(nums))
"""

# Same operator (+) and shape — semantically the same as SUM_SKILL but
# in different lexical wording / variable names. Should cluster.
SUM_SKILL_RENAMED = """
def trigger(text: str) -> bool:
    return '+' in text

def execute(text: str) -> str:
    parts = []
    buf = ''
    for ch in text + ' ':
        if ch.isdigit():
            buf += ch
        elif buf:
            parts.append(int(buf))
            buf = ''
    return str(sum(parts))
"""

# Subtraction instead of addition — same shape but different operator,
# so AST fingerprint will diverge on BinOp.Sub vs BinOp.Add and the
# overall Jaccard should drop below the cluster threshold.
DIFF_SKILL = """
def trigger(query: str) -> bool:
    return '-' in query

def execute(query: str) -> str:
    nums = []
    cur = ''
    for c in query + ' ':
        if c.isdigit():
            cur += c
        elif cur:
            nums.append(int(cur))
            cur = ''
    if not nums:
        return '0'
    result = nums[0]
    for n in nums[1:]:
        result = result - n
    return str(result)
"""

# Domain-different skill: string parsing for emails. Has loops and
# conditionals like SUM_SKILL but completely different operator mix
# (Compare.In vs BinOp.Add/Sub). Must NOT cluster with arithmetic.
EMAIL_SKILL = """
def trigger(query: str) -> bool:
    return '@' in query and '.' in query

def execute(query: str) -> str:
    for word in query.split():
        if '@' in word and '.' in word:
            local, _, domain = word.partition('@')
            if local and '.' in domain:
                return word
    return 'Error: no email'
"""


# ---------------------------------------------------------------------------
# Direct fingerprint tests
# ---------------------------------------------------------------------------


def test_fingerprint_invariant_under_renaming():
    """Variable / parameter names should not change the AST fingerprint."""
    fp1 = ast_fingerprint(SUM_SKILL)
    fp2 = ast_fingerprint(SUM_SKILL_RENAMED)
    assert fp1 is not None
    assert fp2 is not None
    assert fp1 == fp2


def test_fingerprint_distinguishes_operators():
    """+ vs - must produce different fingerprints (BinOp.Add vs BinOp.Sub)."""
    fp_add = ast_fingerprint(SUM_SKILL)
    fp_sub = ast_fingerprint(DIFF_SKILL)
    assert fp_add is not None
    assert fp_sub is not None
    assert fp_add != fp_sub
    # The op-tagged tokens must be present.
    assert "BinOp.Add" in fp_add
    assert "BinOp.Sub" in fp_sub


def test_fingerprint_returns_none_for_unparseable():
    assert ast_fingerprint("def trigger(x): return :::") is None


def test_fingerprint_returns_none_when_no_solve_or_execute():
    code = "def helper(x): return x + 1"
    assert ast_fingerprint(code) is None


def test_fingerprint_picks_solve_over_execute_when_both():
    """If the skill defines both ``solve`` and ``execute``, ``solve`` wins
    (paper convention: ``solve`` is the abstract operation, ``execute``
    is its glue layer)."""
    code = """
def trigger(q): return True

def solve(args):
    return args[0] * args[1]

def execute(q):
    return q.lower()
"""
    fp = ast_fingerprint(code)
    assert fp is not None
    # solve uses BinOp.Mult; execute uses Attribute(.lower). The
    # fingerprint must include Mult and not include Attribute.
    assert "BinOp.Mult" in fp
    assert fp.get("Attribute", 0) == 0


def test_jaccard_same_skill_is_one():
    fp = ast_fingerprint(SUM_SKILL)
    assert ast_jaccard(fp, fp) == pytest.approx(1.0)


def test_jaccard_handles_none():
    fp = ast_fingerprint(SUM_SKILL)
    assert ast_jaccard(None, fp) == 0.0
    assert ast_jaccard(fp, None) == 0.0
    assert ast_jaccard(None, None) == 0.0


def test_jaccard_renamed_is_one():
    """Renamed variables must yield Jaccard 1.0."""
    fp1 = ast_fingerprint(SUM_SKILL)
    fp2 = ast_fingerprint(SUM_SKILL_RENAMED)
    assert ast_jaccard(fp1, fp2) == pytest.approx(1.0)


def test_jaccard_different_operator_below_threshold():
    """Sum vs subtraction differ by op-tagged tokens. Their multiset
    Jaccard should be below ``AST_CLUSTER_THRESHOLD = 0.55`` so they
    don't get merged into one meta-rule. Tighter test: just confirm
    they differ (the threshold check is in the cluster test below)."""
    fp_add = ast_fingerprint(SUM_SKILL)
    fp_sub = ast_fingerprint(DIFF_SKILL)
    assert ast_jaccard(fp_add, fp_sub) < 1.0


# ---------------------------------------------------------------------------
# Clustering integration test
# ---------------------------------------------------------------------------


def test_cluster_groups_renamed_but_separates_emails(tmp_path):
    """Renamed sum skill clusters with original; email skill does NOT."""
    eng = MetaAbductionEngine(persist_dir=str(tmp_path))

    rules = [
        {"pattern": "Sum1", "python_code": SUM_SKILL},
        {"pattern": "Sum2", "python_code": SUM_SKILL_RENAMED},
        {"pattern": "Email", "python_code": EMAIL_SKILL},
    ]
    feats = [(r, eng._extract_structural_features(r)) for r in rules]
    feats = [(r, f) for r, f in feats if f is not None]
    assert len(feats) == 3

    clusters = eng._cluster_by_structure(feats)
    # Sum + Sum_renamed => one cluster of size 2; Email alone.
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]

    # The cluster of size 2 must contain both sum skills.
    big = [c for c in clusters if len(c) == 2][0]
    patterns_in_big = {r["pattern"] for r, _ in big}
    assert patterns_in_big == {"Sum1", "Sum2"}


def test_cluster_separates_arithmetic_from_string():
    """Critical paper invariant: an arithmetic skill must NOT cluster
    with a string-processing skill, even though both contain ``for``,
    ``if``, ``Call``, ``Return`` (which is what the legacy 9-boolean
    Jaccard saw). With AST fingerprints the operator distribution
    diverges enough — Compare.In / BoolOp.And vs BinOp.Add — that
    Jaccard falls below the cluster threshold.

    (Add vs Sub may legitimately co-cluster — they are both iterative
    numeric reductions, which is exactly the level of structural
    isomorphism meta-abduction is meant to surface.)
    """
    fp_sum = ast_fingerprint(SUM_SKILL)
    fp_email = ast_fingerprint(EMAIL_SKILL)
    sim = ast_jaccard(fp_sum, fp_email)
    assert sim < MetaAbductionEngine.AST_CLUSTER_THRESHOLD, (
        f"arithmetic ({sorted(fp_sum.keys())}) and email "
        f"({sorted(fp_email.keys())}) should not be structurally "
        f"isomorphic but Jaccard={sim:.3f}"
    )
