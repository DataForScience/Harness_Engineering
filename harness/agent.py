"""The agent loop and the tool box: the skeleton every later component plugs into.

The loop is deliberately small. Each harness component attaches at one of
these points:

``before_tool``      inspect a call before it runs; return a reason string to block it
``after_tool``       rewrite a tool result before the model sees it
``prepare_context``  rewrite the message list before every model call
``on_final``         post-process the final answer (e.g. verification)
``invoke``           how a tool is actually executed (e.g. with retries/timeouts)
``budget``/``tracer`` accounting and observability
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .llm import ERROR_PREFIX, LLM, ContextWindowExceeded, LLMError, ToolCall
from .mcp_client import MCPClient, ToolResult

DEFAULT_SYSTEM_PROMPT = """You are a data analyst for Trailhead Supply Co.
Answer questions by querying the company's DuckDB database with the tools provided.
Base every number you report on a query result."""


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[[dict], ToolResult]
    annotations: dict = field(default_factory=dict)
    source: str = "local"

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description,
                "input_schema": self.input_schema}


class ToolBox:
    """All tools the model can see: discovered MCP tools plus local ones."""

    def __init__(self, client: MCPClient | None = None):
        self.client = client
        self.tools: dict[str, Tool] = {}
        if client is not None:
            for t in client.list_tools():
                self.tools[t["name"]] = Tool(
                    t["name"], t.get("description", ""), t.get("inputSchema", {}),
                    lambda args, _n=t["name"]: self.client.call_tool(_n, args),
                    t.get("annotations") or {}, source="mcp")

    def add(self, name: str, description: str, input_schema: dict,
            fn: Callable[[dict], ToolResult], **annotations) -> None:
        self.tools[name] = Tool(name, description, input_schema, fn, annotations)

    def remove(self, name: str) -> None:
        self.tools.pop(name, None)

    def specs(self) -> list[dict]:
        return [t.spec() for t in self.tools.values()]

    def call(self, name: str, arguments: dict) -> ToolResult:
        if name not in self.tools:
            return ToolResult(ok=False, text=f"unknown tool {name!r}")
        return self.tools[name].fn(arguments)


class BudgetExceeded(Exception):
    pass


class NullTracer:
    def emit(self, kind: str, **fields: Any) -> None:
        pass


@dataclass
class RunResult:
    answer: str
    status: str                 # completed | max_steps | budget_exceeded | context_overflow | llm_error
    messages: list[dict]
    steps: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    def tool_log(self) -> list[tuple[str, dict, str]]:
        """(tool name, arguments, result content) for every call, in order."""
        calls = {c["id"]: c for m in self.messages if m["role"] == "assistant"
                 for c in m.get("tool_calls") or []}
        return [(calls[m["tool_call_id"]]["name"], calls[m["tool_call_id"]]["arguments"],
                 m["content"]) for m in self.messages
                if m["role"] == "tool" and m.get("tool_call_id") in calls]


class Agent:
    def __init__(self, llm: LLM, tools: ToolBox, system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 max_steps: int = 10,
                 before_tool: list[Callable[[ToolCall], str | None]] = (),
                 after_tool: list[Callable[[ToolCall, ToolResult, str], str]] = (),
                 prepare_context: Callable[[list[dict]], list[dict]] | None = None,
                 on_final: Callable[[str], str] | None = None,
                 invoke: Callable[[str, dict], ToolResult] | None = None,
                 budget=None, tracer=None):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.before_tool = list(before_tool)
        self.after_tool = list(after_tool)
        self.prepare_context = prepare_context
        self.on_final = on_final
        self.invoke = invoke or tools.call
        self.budget = budget
        self.tracer = tracer or NullTracer()

    def execute(self, call: ToolCall) -> str:
        """Run one tool call and return the text the model will see.

        Errors never escape: they come back as ``ERROR: ...`` text so the model
        can react (errors as data)."""
        for check in self.before_tool:
            reason = check(call)
            if reason:
                self.tracer.emit("tool_blocked", tool=call.name, args=call.arguments,
                                 reason=reason)
                return f"{ERROR_PREFIX} blocked by harness: {reason}"
        t0 = time.perf_counter()
        try:
            result = self.invoke(call.name, call.arguments)
        except Exception as e:  # noqa: BLE001 - every failure becomes data
            result = ToolResult(ok=False, text=f"{type(e).__name__}: {e}")
        ms = 1000 * (time.perf_counter() - t0)
        content = result.text if result.ok else f"{ERROR_PREFIX} {result.text}"
        for process in self.after_tool:
            content = process(call, result, content)
        self.tracer.emit("tool_call", tool=call.name, args=call.arguments, ok=result.ok,
                         ms=round(ms, 1), chars=len(content))
        return content

    def run(self, question: str, history: list[dict] | None = None) -> RunResult:
        """Answer ``question``. Pass a previous ``RunResult.messages`` as ``history``
        to continue the same conversation."""
        t0 = time.perf_counter()
        messages = list(history) if history else [{"role": "system", "content": self.system_prompt}]
        messages.append({"role": "user", "content": question})
        res = RunResult(answer="", status="max_steps", messages=messages)
        self.tracer.emit("run_start", question=question)
        try:
            for step in range(1, self.max_steps + 1):
                res.steps = step
                if self.budget:
                    self.budget.check()
                context = self.prepare_context(messages) if self.prepare_context else messages
                try:
                    turn = self.llm.complete(context, self.tools.specs())
                except LLMError:
                    raise
                except Exception as e:  # noqa: BLE001 - a broken model adapter ends the run cleanly
                    raise LLMError(f"model call failed: {type(e).__name__}: {e}") from e
                res.input_tokens += turn.usage.get("input_tokens", 0)
                res.output_tokens += turn.usage.get("output_tokens", 0)
                if self.budget:
                    self.budget.charge_llm(turn.usage)
                self.tracer.emit("llm_call", step=step, tool_calls=len(turn.tool_calls),
                                 **turn.usage)
                messages.append(turn.to_message())
                if not turn.tool_calls:
                    answer = self.on_final(turn.content) if self.on_final else turn.content
                    res.answer, res.status = answer, "completed"
                    break
                for call in turn.tool_calls:
                    if self.budget:
                        self.budget.charge_tool(call.name)
                    res.tool_calls += 1
                    messages.append({"role": "tool", "tool_call_id": call.id,
                                     "name": call.name, "content": self.execute(call)})
            else:
                res.answer = f"Stopped after {self.max_steps} steps without an answer."
        except BudgetExceeded as e:
            res.status, res.answer = "budget_exceeded", f"Stopped: {e}"
        except ContextWindowExceeded as e:
            res.status, res.answer = "context_overflow", f"Stopped: {e}"
        except LLMError as e:
            res.status, res.answer = "llm_error", f"Stopped: {e}"
        res.seconds = time.perf_counter() - t0
        self.tracer.emit("run_end", status=res.status, steps=res.steps,
                         tool_calls=res.tool_calls)
        return res
