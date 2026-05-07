"""Subprocess sandbox tests (Tier A4).

These exercise the actual end-to-end path: AST validation in the
parent, then subprocess execution with rlimits and timeout. They are
slower than the in-process tests (each spawns a real Python child),
so we keep the count small but cover the critical invariants:

1. Happy path: a valid skill returns its computed output.
2. SecurityError is raised in the parent (no subprocess spawn) when
   AST validation rejects forbidden constructs.
3. Wall-clock timeout kills the child and surfaces
   SubprocessSandboxError.
4. Skill runtime errors surface as SubprocessSandboxError, not as a
   silent string.
5. ``trigger``-only mode returns the boolean as a string.
6. The subprocess sandbox tolerates the same valid skills the
   in-process sandbox accepts (no spurious AST regressions).
"""

from __future__ import annotations

import os
import sys

import pytest

from nare.execution.sandboxes.base import SecurityError
from nare.execution.local import (
    SubprocessSandboxError,
    safe_call_trigger_subprocess,
    safe_execute_subprocess,
)


GOOD_SKILL = """
def trigger(query: str) -> bool:
    return any(c.isdigit() for c in query)

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


def test_subprocess_happy_path():
    out = safe_execute_subprocess(GOOD_SKILL, "2 + 5 + 7")
    assert out == "14"


def test_subprocess_trigger_false_returns_error_string():
    out = safe_execute_subprocess(GOOD_SKILL, "no digits here")
    assert out.startswith("Error: trigger")


def test_subprocess_trigger_only_mode():
    """`trigger` mode should return 'True' or 'False' as a string,
    matching the symmetry with in-process safe_call_trigger."""
    assert safe_call_trigger_subprocess(GOOD_SKILL, "has 1 digit") is True
    assert safe_call_trigger_subprocess(GOOD_SKILL, "no digits") is False


def test_subprocess_blocks_open_at_ast_layer():
    """`open()` is rejected by the AST validator BEFORE we ever fork."""
    bad = """
def trigger(query: str) -> bool:
    return True

def execute(query: str) -> str:
    open('/tmp/ohno', 'w').write('escape')
    return 'done'
"""
    with pytest.raises(SecurityError):
        safe_execute_subprocess(bad, "anything")


def test_subprocess_blocks_dunder_at_ast_layer():
    bad = """
def trigger(query: str) -> bool:
    return True

def execute(query: str) -> str:
    return str(().__class__.__bases__[0].__subclasses__())
"""
    with pytest.raises(SecurityError):
        safe_execute_subprocess(bad, "anything")


def test_subprocess_runtime_error_surfaces():
    """An exception inside execute() must come back as
    SubprocessSandboxError, not be silently swallowed."""
    crashing = """
def trigger(query: str) -> bool:
    return True

def execute(query: str) -> str:
    raise ValueError('boom')
"""
    with pytest.raises(SubprocessSandboxError) as excinfo:
        safe_execute_subprocess(crashing, "x")
    assert "execute" in str(excinfo.value).lower()
    assert "valueerror" in str(excinfo.value).lower() or "boom" in str(excinfo.value).lower()


@pytest.mark.skipif(
    os.name != "posix",
    reason="wall-clock timeout requires reliable signal delivery (POSIX)",
)
def test_subprocess_timeout_enforced():
    """A skill that loops forever must be killed within the timeout."""
    looper = """
def trigger(query: str) -> bool:
    return True

def execute(query: str) -> str:
    while True:
        pass
"""
    with pytest.raises(SubprocessSandboxError) as excinfo:
        safe_execute_subprocess(looper, "x", timeout=1.0, cpu_limit_seconds=2)
    msg = str(excinfo.value).lower()
    # Either the wall-clock timeout fires, or the rlimit-cpu kills it
    # first (returncode != 0). Both are acceptable.
    assert ("budget" in msg) or ("exited" in msg) or ("output" in msg)


def test_subprocess_string_output_is_stringified():
    """Skill that returns an int gets stringified (mirrors safe_execute)."""
    code = """
def trigger(query: str) -> bool:
    return True

def execute(query: str):
    return 42
"""
    out = safe_execute_subprocess(code, "anything")
    assert out == "42"
