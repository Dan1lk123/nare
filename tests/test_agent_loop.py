"""Unit tests for the NARE agent tool-calling loop.

Covers:
  - tool registry validation and dispatch
  - parse_turn() with well-formed and malformed inputs
  - end-to-end loop with a fake LLM (multi-tool, final answer, budget)
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from nare.agents.events import (
    EventBus,
    PlanProposed,
    TaskFinished,
    TaskStarted,
    Thought,
    ToolEnd,
    ToolStart,
)
from nare.agents.loop import (
    AgentLoop,
    Budget,
    ParsedTurn,
    build_loop,
    parse_turn,
)
from nare.agents.tools import build_default_registry
from nare.agents.tools.base import Tool, ToolError, ToolParam, ToolRegistry, ToolResult


# ─────────────────────────────────────────────────────────────────────
# Tool registry
# ─────────────────────────────────────────────────────────────────────


def test_tool_validate_required():
    reg = ToolRegistry()
    reg.register(Tool(
        name="echo",
        description="echo something",
        parameters=[ToolParam("text", "string", "what to echo")],
        run=lambda text: ToolResult(ok=True, body=text),
    ))
    with pytest.raises(ToolError):
        reg.tools["echo"].validate({})  # missing 'text'


def test_tool_call_returns_string_wraps_to_result():
    reg = ToolRegistry()
    reg.register(Tool(
        name="echo",
        description="echo",
        parameters=[ToolParam("text", "string", "what to echo")],
        run=lambda text: text,  # returns plain str
    ))
    r = reg.call("echo", {"text": "hi"})
    assert r.ok and r.body == "hi"


def test_default_registry_has_core_tools():
    reg = build_default_registry()
    for name in (
        "read_file",
        "write_file",
        "edit_file",
        "bash",
        "grep",
        "list_dir",
        "update_todos",
    ):
        assert name in reg.tools


def test_update_todos_normalizes_and_emits_event():
    """`update_todos` must normalize states and produce TodoUpdated."""
    from nare.agents.events import EventBus, TodoUpdated
    from nare.agents.loop import build_loop

    bus = EventBus()
    seen: list = []
    bus.subscribe(lambda ev: seen.append(ev))

    reg = build_default_registry()
    r = reg.call("update_todos", {
        "items": [
            {"text": "Plan task", "state": "done"},
            {"text": "Implement", "state": "in_progress"},
            {"text": "Test", "state": "pending"},
            {"text": "Lowercase", "state": "DONE"},   # case-folded
            {"text": "Bad state", "state": "weird"},  # falls back to pending
        ],
    })
    assert r.ok
    items = r.meta["_todo_items"]
    assert items[0] == {"text": "Plan task", "state": "done"}
    assert items[3] == {"text": "Lowercase", "state": "done"}
    assert items[4] == {"text": "Bad state", "state": "pending"}
    assert "5 todos" in r.summary
    counts = r.meta["counts"]
    assert counts["done"] == 2 and counts["in_progress"] == 1 and counts["pending"] == 2

    # End-to-end: when the loop dispatches update_todos it must also
    # emit a TodoUpdated event.
    loop = build_loop()
    loop.bus = bus
    fake_outputs = iter([
        '<tool_call>{"name": "update_todos", "args": {"items": '
        '[{"text": "a", "state": "done"}, {"text": "b", "state": "pending"}]}}</tool_call>',
        '<final_answer>todos updated</final_answer>',
    ])
    loop.llm_call = lambda prompt: next(fake_outputs)
    out = loop.run("update my todos")
    assert out.ok and out.final_answer == "todos updated"
    assert any(isinstance(e, TodoUpdated) for e in seen)


def test_builtin_read_write_round_trip():
    with tempfile.TemporaryDirectory() as td:
        reg = build_default_registry(working_dir=td)
        w = reg.call("write_file", {"path": "x.txt", "content": "hello\nworld\n"})
        assert w.ok and "Wrote 2 lines" in (w.summary or "")
        r = reg.call("read_file", {"path": "x.txt"})
        assert r.ok and r.body == "hello\nworld\n"


def test_builtin_edit_file_unique_match():
    with tempfile.TemporaryDirectory() as td:
        reg = build_default_registry(working_dir=td)
        reg.call("write_file", {"path": "a.py", "content": "x = 1\ny = 2\n"})
        e = reg.call("edit_file", {"path": "a.py", "old": "y = 2", "new": "y = 99"})
        assert e.ok and e.meta["additions"] == 1 and e.meta["deletions"] == 1
        assert (reg.call("read_file", {"path": "a.py"}).body
                == "x = 1\ny = 99\n")


def test_builtin_edit_file_ambiguous_without_replace_all():
    with tempfile.TemporaryDirectory() as td:
        reg = build_default_registry(working_dir=td)
        reg.call("write_file", {"path": "a.py", "content": "x = 1\nx = 1\n"})
        e = reg.call("edit_file", {"path": "a.py", "old": "x = 1", "new": "x = 99"})
        assert not e.ok and "appears" in (e.error or "")


def test_builtin_bash_done_for_empty_output():
    reg = build_default_registry()
    r = reg.call("bash", {"command": "true"})
    assert r.ok and r.summary == "Done"


def test_builtin_bash_exit_code():
    reg = build_default_registry()
    r = reg.call("bash", {"command": "false"})
    assert not r.ok and r.summary == "exit 1"


# ─────────────────────────────────────────────────────────────────────
# parse_turn
# ─────────────────────────────────────────────────────────────────────


def test_parse_turn_final_answer():
    p = parse_turn("<final_answer>hello</final_answer>")
    assert p.kind == "final_answer" and p.answer == "hello"


def test_parse_turn_tool_call_well_formed():
    raw = '<tool_call>{"name": "read_file", "args": {"path": "a.py"}}</tool_call>'
    p = parse_turn(raw)
    assert p.kind == "tool_call"
    assert p.tool_name == "read_file"
    assert p.tool_args == {"path": "a.py"}


def test_parse_turn_malformed_no_block():
    p = parse_turn("just some text without any tag")
    assert p.kind == "malformed" and "no <tool_call>" in (p.error or "")


def test_parse_turn_malformed_invalid_json():
    p = parse_turn("<tool_call>{not valid json}</tool_call>")
    assert p.kind == "malformed" and "JSON parse error" in (p.error or "")


def test_parse_turn_malformed_missing_name():
    p = parse_turn('<tool_call>{"args": {}}</tool_call>')
    assert p.kind == "malformed" and "name" in (p.error or "")


def test_parse_turn_final_answer_takes_precedence():
    raw = (
        "<tool_call>{\"name\":\"x\",\"args\":{}}</tool_call>"
        "<final_answer>done</final_answer>"
    )
    p = parse_turn(raw)
    assert p.kind == "final_answer"


# ─────────────────────────────────────────────────────────────────────
# Loop end-to-end (fake LLM)
# ─────────────────────────────────────────────────────────────────────


def _capture_events(bus: EventBus):
    events = []
    bus.subscribe(events.append)
    return events


def test_loop_single_tool_then_final():
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "h.txt"), "w") as f:
            f.write("hi\n")
        turns = iter([
            '<tool_call>{"name": "read_file", "args": {"path": "h.txt"}}</tool_call>',
            "<final_answer>It says hi.</final_answer>",
        ])
        loop = build_loop(working_dir=td)
        loop.triage = None
        loop.planner = None
        loop.llm_call = lambda _: next(turns)
        events = _capture_events(loop.bus)

        r = loop.run("what's in h.txt")
        assert r.ok
        assert r.final_answer == "It says hi."
        assert r.iterations == 2
        assert r.stop_reason == "final_answer"

        kinds = [type(e).__name__ for e in events]
        assert kinds[0] == "TaskStarted"
        assert "ToolStart" in kinds and "ToolEnd" in kinds
        assert kinds[-1] == "TaskFinished"


def test_loop_multi_tool_chain():
    with tempfile.TemporaryDirectory() as td:
        for name in ("a.py", "b.py"):
            with open(os.path.join(td, name), "w") as f:
                f.write("print('x')\n")
        turns = iter([
            '<tool_call>{"name": "list_dir", "args": {}}</tool_call>',
            '<tool_call>{"name": "grep", "args": {"pattern": "print", "glob": "*.py"}}</tool_call>',
            "<final_answer>Two files print.</final_answer>",
        ])
        loop = build_loop(working_dir=td)
        loop.triage = None
        loop.planner = None
        loop.llm_call = lambda _: next(turns)
        r = loop.run("count prints")
        assert r.ok and r.iterations == 3 and r.final_answer.startswith("Two files")


def test_loop_budget_max_iterations_stops_loop():
    loop = build_loop(working_dir=".")
    loop.triage = None
    loop.planner = None
    loop.budget = Budget(max_iterations=2, max_tokens=10**9, max_wall_clock=60)
    loop.llm_call = lambda _: '<tool_call>{"name": "list_dir", "args": {}}</tool_call>'
    r = loop.run("forever")
    assert not r.ok
    assert r.iterations == 2
    assert "budget_exhausted" in r.stop_reason
    assert r.final_answer  # graceful partial answer


def test_loop_malformed_retry_then_final():
    attempts = {"n": 0}

    def fake(prompt):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return "naked text without any tags"
        return "<final_answer>recovered</final_answer>"

    loop = build_loop(working_dir=".")
    loop.triage = None
    loop.planner = None
    loop.llm_call = fake
    r = loop.run("smoke")
    assert r.ok and r.iterations == 2 and r.final_answer == "recovered"


def test_loop_emits_plan_for_edit_intent():
    pytest.importorskip("nare.agents.planning")
    # The real PlanningAgent calls the LLM, which we don't want here.
    # Inject a fake planner instead.
    class FakePlanner:
        def generate_plan(self, task, repo_map=None):
            return {
                "plan_steps": ["read file", "edit file"],
                "target_files": ["a.py"],
                "complexity": "trivial",
            }

    class FakeTriage:
        def classify(self, q):
            return "EDIT"

    loop = build_loop(working_dir=".")
    loop.triage = FakeTriage()
    loop.planner = FakePlanner()
    loop.llm_call = lambda _: "<final_answer>ok</final_answer>"
    events = _capture_events(loop.bus)
    r = loop.run("edit something")
    assert r.ok
    assert any(isinstance(e, PlanProposed) and e.steps for e in events)


def test_loop_unknown_tool_returns_error_observation():
    """Loop should keep going and let the LLM recover from a bad tool name."""
    turns = iter([
        '<tool_call>{"name": "no_such_tool", "args": {}}</tool_call>',
        "<final_answer>I gave up.</final_answer>",
    ])
    loop = build_loop(working_dir=".")
    loop.triage = None
    loop.planner = None
    loop.llm_call = lambda _: next(turns)
    r = loop.run("invoke ghost")
    assert r.ok and r.iterations == 2


# ─────────────────────────────────────────────────────────────────────
# Final-answer streaming filter
# ─────────────────────────────────────────────────────────────────────


def test_final_answer_streamer_emits_only_inside_block():
    from nare.agents.loop import _make_final_answer_streamer

    out: list[str] = []
    push = _make_final_answer_streamer(out.append)
    for d in [
        "Some preamble.\n<final_",
        "answer>\nHello",
        " world!\n",
        "How are you?\n",
        "</final_answer> trailing junk",
    ]:
        push(d)
    assert "".join(out) == "Hello world!\nHow are you?\n"


def test_final_answer_streamer_silences_tool_calls():
    from nare.agents.loop import _make_final_answer_streamer

    out: list[str] = []
    push = _make_final_answer_streamer(out.append)
    for d in ["<tool_call>", '{"name":"list_dir"}', "</tool_call>"]:
        push(d)
    assert "".join(out) == ""


def test_final_answer_streamer_handles_split_close_tag():
    from nare.agents.loop import _make_final_answer_streamer

    out: list[str] = []
    push = _make_final_answer_streamer(out.append)
    for d in ["<final_answer>The answer is 42.", "</fin", "al_answer>"]:
        push(d)
    assert "".join(out) == "The answer is 42."
