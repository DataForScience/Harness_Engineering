"""Run supervision: timeouts, retries, budgets, traces and fault injection.

``ResilientInvoker``  executes tool calls with a deadline, classified retries,
                      exponential backoff and server restarts
``Budget``            hard limits on steps, tool calls, tokens and wall-clock time
``Tracer``            structured, append-only run events (JSONL-ready)
``FaultInjector``     wraps an invoker and breaks it on purpose, deterministically
"""

from __future__ import annotations

import json
import random
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .agent import BudgetExceeded
from .mcp_client import MCPClient, MCPConnectionError, MCPTimeout, ToolResult


# --- error classification --------------------------------------------------------------

TRANSIENT_MARKERS = ("io error", "could not set lock", "connection reset", "temporarily",
                     "try again", "503", "502", "rate limit")


def classify_failure(error: Exception | ToolResult) -> str:
    """timeout | transient | permanent. Only the first two are worth retrying."""
    if isinstance(error, MCPTimeout) or isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, (MCPConnectionError, ConnectionError, BrokenPipeError)):
        return "transient"
    text = error.text if isinstance(error, ToolResult) else str(error)
    if any(m in text.lower() for m in TRANSIENT_MARKERS):
        return "transient"
    return "permanent"


# --- tracing -----------------------------------------------------------------------------

class Tracer:
    """Collects structured events; one dict per event, ready for JSONL or pandas."""

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.t0 = time.perf_counter()
        self.events: list[dict] = []

    def emit(self, kind: str, **fields: Any) -> None:
        self.events.append({"run_id": self.run_id,
                            "t": round(time.perf_counter() - self.t0, 4),
                            "kind": kind, **fields})

    def to_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            for e in self.events:
                f.write(json.dumps(e, default=str) + "\n")
        return path

    def df(self):
        import pandas as pd
        return pd.DataFrame(self.events)

    def counts(self) -> Counter:
        return Counter(e["kind"] for e in self.events)


# --- resilient execution ----------------------------------------------------------------------

class ResilientInvoker:
    """Execute a tool with a deadline, retry what is worth retrying, restart what is broken.

    Permanent errors (bad SQL, unknown table) are returned to the model at once:
    retrying them only wastes budget. The model re-plans instead."""

    def __init__(self, client: MCPClient, call: Callable[[str, dict], ToolResult] | None = None,
                 timeout: float = 10.0, max_retries: int = 2, backoff: float = 0.2,
                 tracer: Tracer | None = None, seed: int = 0):
        self.client = client
        client.timeout = timeout
        self.call = call or (lambda name, args: client.call_tool(name, args, timeout=self.timeout))
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.tracer = tracer
        self.rng = random.Random(seed)

    def _emit(self, kind: str, **f) -> None:
        if self.tracer:
            self.tracer.emit(kind, **f)

    def __call__(self, name: str, arguments: dict) -> ToolResult:
        last: Exception | ToolResult | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                delay = self.backoff * 2 ** (attempt - 1) * (1 + self.rng.random() / 2)
                self._emit("retry", tool=name, attempt=attempt, delay=round(delay, 3),
                           cause=classify_failure(last))
                time.sleep(delay)
            try:
                if not self.client.alive:
                    self._emit("restart", reason="server not running")
                    self.client.restart()
                result = self.call(name, arguments)
            except Exception as e:  # noqa: BLE001
                kind = classify_failure(e)
                self._emit("tool_failure", tool=name, attempt=attempt, cls=kind,
                           error=f"{type(e).__name__}: {e}")
                last = e
                if kind == "permanent":
                    break
                if kind == "timeout":   # a hung server may never answer: start over
                    self._emit("restart", reason="timeout")
                    self.client.restart()
                continue
            if result.ok and result.data is None and result.text.lstrip()[:1] not in ("{", "["):
                last = ToolResult(False, f"malformed tool response: {result.text[:80]!r}")
                self._emit("tool_failure", tool=name, attempt=attempt, cls="transient",
                           error=last.text)
                continue
            if result.ok or classify_failure(result) == "permanent":
                return result
            self._emit("tool_failure", tool=name, attempt=attempt, cls="transient",
                       error=result.text[:200])
            last = result
        if isinstance(last, ToolResult):
            return ToolResult(False, f"{last.text} (gave up after {attempt + 1} attempts)")
        return ToolResult(False, f"{type(last).__name__}: {last} "
                                 f"(gave up after {attempt + 1} attempts)")


class ReplanAdvisor:
    """``after_tool`` hook: after repeated failures, tell the model to change course."""

    def __init__(self, threshold: int = 2):
        self.threshold = threshold
        self.consecutive = 0

    def __call__(self, call, result: ToolResult, content: str) -> str:
        self.consecutive = 0 if result.ok else self.consecutive + 1
        if self.consecutive >= self.threshold:
            return (content + f"\n[harness] {self.consecutive} tool calls in a row have failed. "
                    "Do not repeat the same call. Try a simpler approach, or answer with what "
                    "you have and say what is missing.")
        return content


# --- fault injection ------------------------------------------------------------------------------

FAULTS = ("timeout", "transient", "crash", "malformed", "oversized", "slow", "permanent")


class FaultInjector:
    """Wraps a tool-call function and injects faults on chosen calls.

    ``plan`` maps the call number (1-based) to a fault name; ``rate`` adds random
    faults with a fixed seed so every run is reproducible."""

    def __init__(self, call: Callable[[str, dict], ToolResult], client: MCPClient | None = None,
                 plan: dict[int, str] | None = None, rate: float = 0.0,
                 faults: tuple[str, ...] = FAULTS, seed: int = 0, slow_seconds: float = 0.5):
        self.inner = call
        self.client = client
        self.plan = dict(plan or {})
        self.rate = rate
        self.faults = faults
        self.rng = random.Random(seed)
        self.slow_seconds = slow_seconds
        self.n = 0
        self.injected: list[tuple[int, str]] = []

    def __call__(self, name: str, arguments: dict) -> ToolResult:
        self.n += 1
        fault = self.plan.get(self.n)
        if fault is None and self.rate and self.rng.random() < self.rate:
            fault = self.rng.choice(self.faults)
        if fault is None:
            return self.inner(name, arguments)
        self.injected.append((self.n, fault))
        if fault == "timeout":
            raise MCPTimeout(f"{name} timed out (injected)")
        if fault == "transient":
            raise MCPConnectionError("connection reset by peer (injected)")
        if fault == "crash":
            if self.client and self.client.proc:
                self.client.proc.kill()
                self.client.proc.wait()
            return self.inner(name, arguments)   # fails for real: the server is dead
        if fault == "malformed":
            return ToolResult(True, "<html><body>502 Bad Gateway</body></html>")
        if fault == "oversized":
            rows = [[i, "x" * 500] for i in range(2000)]
            data = {"success": True, "columns": ["id", "blob"], "columnTypes": ["BIGINT", "VARCHAR"],
                    "rows": rows, "rowCount": len(rows)}
            return ToolResult(True, json.dumps(data), data)
        if fault == "slow":
            time.sleep(self.slow_seconds)
            return self.inner(name, arguments)
        if fault == "permanent":
            return ToolResult(False, 'Catalog Error: Table with name ordrs does not exist! '
                                     'Did you mean "orders"?')
        raise ValueError(f"unknown fault {fault!r}")


# --- budgets -------------------------------------------------------------------------------

@dataclass
class Budget:
    max_tool_calls: int = 12
    max_tokens: int = 60_000
    max_seconds: float = 60.0
    max_calls_per_tool: int = 8
    tool_calls: int = 0
    tokens: int = 0
    per_tool: Counter = field(default_factory=Counter)
    started: float = field(default_factory=time.perf_counter)

    def reset(self) -> None:
        self.tool_calls, self.tokens = 0, 0
        self.per_tool = Counter()
        self.started = time.perf_counter()

    def check(self) -> None:
        elapsed = time.perf_counter() - self.started
        if elapsed > self.max_seconds:
            raise BudgetExceeded(f"time budget of {self.max_seconds:.0f}s exhausted")
        if self.tokens > self.max_tokens:
            raise BudgetExceeded(f"token budget of {self.max_tokens:,} exhausted ({self.tokens:,})")

    def charge_llm(self, usage: dict) -> None:
        self.tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        self.check()

    def charge_tool(self, name: str) -> None:
        self.tool_calls += 1
        self.per_tool[name] += 1
        if self.tool_calls > self.max_tool_calls:
            raise BudgetExceeded(f"tool-call budget of {self.max_tool_calls} exhausted")
        if self.per_tool[name] > self.max_calls_per_tool:
            raise BudgetExceeded(f"{name} called more than {self.max_calls_per_tool} times")
        self.check()

    def remaining(self) -> dict:
        return {"tool_calls": self.max_tool_calls - self.tool_calls,
                "tokens": self.max_tokens - self.tokens,
                "seconds": round(self.max_seconds - (time.perf_counter() - self.started), 1)}
