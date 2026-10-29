"""Context management: keep the model inside a token budget without losing facts.

Four tools for one job:

``summarize_result``  turn a big query result into a compact, honest summary
``ResultStore``       keep the full result outside the context, behind a handle
``ContextCompactor``  shrink the conversation when it grows past a budget
``MemoryStore``       tiered memory: durable facts recalled into the prompt
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .agent import ToolBox
from .llm import ToolCall, count_message_tokens, estimate_tokens
from .mcp_client import ToolResult


# --- compact result summaries ----------------------------------------------------------


def _is_num(x) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        return True
    try:
        float(x)
        return isinstance(x, str) and bool(re.fullmatch(r"-?\d+(\.\d+)?", x))
    except (TypeError, ValueError):
        return False


def column_profile(name: str, ctype: str, values: list) -> dict:
    vals = [v for v in values if v is not None]
    prof = {"name": name, "type": ctype, "nulls": len(values) - len(vals)}
    if not vals:
        return prof
    if all(_is_num(v) for v in vals):
        nums = [float(v) for v in vals]
        prof.update(min=min(nums), max=max(nums), mean=round(sum(nums) / len(nums), 2))
    else:
        strs = [str(v) for v in vals]
        distinct = sorted(set(strs))
        prof["distinct"] = len(distinct)
        if len(distinct) <= 8:
            prof["values"] = distinct
        elif max(map(len, strs)) <= 12:
            prof.update(min=min(strs), max=max(strs))
        else:
            prof["avg_length"] = round(sum(map(len, strs)) / len(strs))
    return prof


def summarize_result(data: dict, handle: str, preview_rows: int = 3,
                     max_cell: int = 80) -> dict:
    """Compact stand-in for a query result: shape, column profiles, a few rows."""
    cols, rows = data.get("columns", []), data.get("rows", [])
    types = data.get("columnTypes", ["?"] * len(cols))
    clip = lambda v: v[:max_cell] + "..." if isinstance(v, str) and len(v) > max_cell else v
    summary = {
        "handle": handle,
        "row_count": len(rows),
        "columns": [column_profile(c, t, [r[i] for r in rows])
                    for i, (c, t) in enumerate(zip(cols, types))],
        "preview": [[clip(v) for v in r] for r in rows[:preview_rows]],
        "note": f"Full result stored as {handle!r}; call expand_result to page through it. "
                "Prefer aggregating in SQL over reading rows.",
    }
    if data.get("truncated"):
        summary["truncated"] = True
        summary["note"] = ("WARNING: the server truncated this result, so row_count is NOT "
                           "the true total. Use COUNT(*)/aggregates in SQL. " + summary["note"])
    return summary


# --- handles: keep full results out of the context --------------------------------------

class ResultStore:
    """Full results live here, out of the context window, behind short handles."""

    def __init__(self):
        self.results: dict[str, dict] = {}

    def put(self, data: dict) -> str:
        handle = f"r{len(self.results) + 1}"
        self.results[handle] = data
        return handle

    def page(self, handle: str, offset: int = 0, limit: int = 10,
             max_chars: int = 3000) -> dict:
        if handle not in self.results:
            return {"error": f"unknown handle {handle!r}"}
        data = self.results[handle]
        rows, out = data["rows"][offset:offset + limit], []
        for r in rows:  # stop early rather than blow the budget
            if len(json.dumps(out + [r], default=str)) > max_chars:
                break
            out.append(r)
        return {"handle": handle, "columns": data["columns"], "offset": offset,
                "rows": out, "returned": len(out), "total": len(data["rows"]),
                "next_offset": offset + len(out) if offset + len(out) < len(data["rows"]) else None}

    def register_tool(self, toolbox: ToolBox) -> None:
        def expand(args: dict) -> ToolResult:
            page = self.page(args["handle"], int(args.get("offset", 0)),
                             min(int(args.get("limit", 10)), 50))
            return ToolResult(ok="error" not in page, text=json.dumps(page, default=str),
                              data=page)
        toolbox.add("expand_result",
                    "Page through a stored query result by handle (max 50 rows per call).",
                    {"type": "object", "additionalProperties": False, "required": ["handle"],
                     "properties": {"handle": {"type": "string"},
                                    "offset": {"type": "integer"},
                                    "limit": {"type": "integer"}}},
                    expand, readOnlyHint=True)


# --- wiring it in: an after_tool hook --------------------------------------------------

class ResultCompactor:
    """``after_tool`` hook: results over ``max_chars`` become summary + handle."""

    def __init__(self, store: ResultStore, max_chars: int = 1500):
        self.store = store
        self.max_chars = max_chars

    def __call__(self, call: ToolCall, result: ToolResult, content: str) -> str:
        data = result.data
        if not result.ok or not data or "rows" not in data or len(content) <= self.max_chars:
            return content
        handle = self.store.put(data)
        return json.dumps(summarize_result(data, handle), default=str)


# --- conversation compaction ------------------------------------------------------------

class ContextCompactor:
    """``prepare_context`` hook that keeps the prompt under ``budget`` tokens.

    Stage 1: elide the content of old tool results (keeps call/result pairing).
    Stage 2: fold the oldest exchanges into a single summary message.
    The system prompt, the question and the last ``keep_last`` messages are kept."""

    def __init__(self, budget: int = 4000, keep_last: int = 4):
        self.budget = budget
        self.keep_last = keep_last
        self.log: list[dict] = []

    def __call__(self, messages: list[dict]) -> list[dict]:
        before = count_message_tokens(messages)
        if before <= self.budget:
            return messages
        head, body = messages[:2], [dict(m) for m in messages[2:]]
        recent_start = max(0, len(body) - self.keep_last)
        users = [i for i, m in enumerate(body) if m["role"] == "user"]
        if users:  # never fold away the question currently being answered
            recent_start = min(recent_start, users[-1])
        # Stage 1: elide old tool outputs, oldest first.
        for m in body[:recent_start]:
            if count_message_tokens(head + body) <= self.budget:
                break
            if m["role"] == "tool" and not m["content"].startswith("[elided"):
                handle = re.search(r'"handle": "(r\d+)"', m["content"])
                m["content"] = (f"[elided {estimate_tokens(m['content'])} tokens"
                                + (f"; full result in {handle.group(1)}" if handle else "") + "]")
        # Stage 2: fold old exchanges into one summary message.
        if count_message_tokens(head + body) > self.budget and recent_start > 0:
            old, body = body[:recent_start], body[recent_start:]
            # never start the kept window on a tool result whose call was folded away
            while body and body[0]["role"] == "tool":
                old.append(body.pop(0))
            done = [f"{c['name']}({json.dumps(c['arguments'])[:120]})"
                    for m in old if m["role"] == "assistant" for c in m.get("tool_calls") or []]
            body = [{"role": "user", "content": "Summary of earlier steps: called "
                     + "; ".join(done) + ". Their results were elided to save context."}] + body
        out = head + body
        self.log.append({"before": before, "after": count_message_tokens(out)})
        return out


# --- tiered memory -----------------------------------------------------------------------

_WORD = re.compile(r"[a-z_]{3,}")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


@dataclass
class Memory:
    text: str
    source: str            # who asserted it: "human", "verified", "agent"
    tags: list[str] = field(default_factory=list)


class MemoryStore:
    """Long-term memory: small, curated facts that survive across runs.

    Facts are recalled by word overlap with the question and placed in the
    system prompt. Only trusted sources may write here; anything the agent read
    inside data is *not* a trusted source."""

    TRUSTED = {"human", "verified"}

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.items: list[Memory] = []
        if self.path and self.path.exists():
            self.items = [Memory(**m) for m in json.loads(self.path.read_text())]

    def add(self, text: str, source: str, tags: list[str] = ()) -> bool:
        if source not in self.TRUSTED:
            return False
        if all(m.text != text for m in self.items):
            self.items.append(Memory(text, source, list(tags)))
            self.save()
        return True

    def recall(self, query: str, k: int = 3) -> list[Memory]:
        q = _words(query)
        scored = [(len(q & (_words(m.text) | set(m.tags))), i, m)
                  for i, m in enumerate(self.items)]
        return [m for s, _, m in sorted(scored, key=lambda x: (-x[0], x[1])) if s > 0][:k]

    def save(self) -> None:
        if self.path:
            self.path.write_text(json.dumps([m.__dict__ for m in self.items], indent=1))

    def system_prompt(self, base: str, question: str, k: int = 3) -> str:
        facts = self.recall(question, k)
        if not facts:
            return base
        return base + "\n\nKnown facts about this database:\n" + "\n".join(
            f"- {m.text}" for m in facts)
