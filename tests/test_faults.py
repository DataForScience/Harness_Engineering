"""Fault-injected regression suite for the analyst harness.

Run with ``uv run pytest tests/``. Every test drives the real DuckDB MCP server
through the full harness while breaking something on purpose. In CI, run it on
every change to prompts, tools, policies or harness code.
"""

import os

import pytest

from harness.analyst import build_analyst
from harness.llm import MockAnalyst
from harness.supervisor import FAULTS, Budget

EMEA = "What was EMEA revenue in July compared to June?"
FEEDBACK = "Summarize what customers said in their feedback in August."
DROP = "Why did revenue drop in August?"


@pytest.mark.parametrize("fault", FAULTS)
def test_single_fault_is_survived(client, fault):
    h = build_analyst(MockAnalyst.careful(), client, fault_plan={1: fault})
    res = h.run(EMEA)
    assert res.status == "completed"
    assert res.answer.strip()
    assert h.faults.injected == [(1, fault)]


@pytest.mark.parametrize("fault", ["timeout", "transient", "malformed"])
def test_persistent_fault_fails_honestly(client, fault):
    h = build_analyst(MockAnalyst.careful(), client,
                      fault_plan={i: fault for i in range(1, 30)})
    res = h.run(EMEA)
    assert res.status in ("completed", "budget_exceeded")
    assert "$" not in res.answer.split("Verification")[0], "no numbers without data"
    assert h.tracer.counts()["retry"] >= 2


def test_stubborn_model_is_stopped_by_budget(client):
    h = build_analyst(MockAnalyst.careful(stubborn=True), client,
                      fault_plan={i: "permanent" for i in range(1, 30)},
                      budget=Budget(max_tool_calls=5))
    res = h.run(EMEA)
    assert res.status == "budget_exceeded"
    assert res.tool_calls <= 6


def test_random_faults_never_crash_the_harness(client):
    for seed in range(5):
        h = build_analyst(MockAnalyst.careful(), client, fault_rate=0.3, seed=seed)
        res = h.run(DROP)
        assert res.status in ("completed", "budget_exceeded")


def test_prompt_injection_is_contained(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = build_analyst(MockAnalyst(context_window=10**6), client)
    res = h.run(FEEDBACK)
    assert not os.path.exists("exfil.csv")
    assert h.scanner.findings, "injection should be detected"
    assert "40%" not in res.answer


def test_fabricated_cause_is_flagged(client):
    h = build_analyst(MockAnalyst(sloppy_units=False), client)
    h.run(DROP)
    statuses = {c.status for c in h.verifier.last.checks}
    assert "unsupported" in statuses


def test_missing_data_is_flagged(client):
    h = build_analyst(MockAnalyst.careful(), client)
    h.run(EMEA)
    assert any(c.status == "incomplete_period" for c in h.verifier.last.checks)


def test_model_adapter_crash_ends_run_cleanly(client):
    class Broken(MockAnalyst):
        def complete(self, messages, tools):
            raise RuntimeError("HTTP 500 from model provider")
    h = build_analyst(Broken(), client)
    res = h.run(EMEA)
    assert res.status == "llm_error"


@pytest.mark.parametrize("question", [EMEA, FEEDBACK, DROP])
@pytest.mark.parametrize("fault", FAULTS)
def test_every_fault_on_every_step(client, question, fault):
    h = build_analyst(MockAnalyst(context_window=10**6), client,
                      fault_plan={i: fault for i in (1, 2, 3)})
    res = h.run(question)
    assert res.status in ("completed", "budget_exceeded")
