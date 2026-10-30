"""Model interface plus two offline models: ``ScriptedLLM`` and ``MockAnalyst``.

The harness only depends on ``LLM.complete(messages, tools) -> AssistantTurn``.
Any real provider can be wrapped to satisfy it. The workshop uses
``MockAnalyst``: a deterministic, rule-based stand-in that *behaves like* a
capable but careless model. It plans SQL, reads results and writes reports, and
it reproduces common failure modes on demand:

``fabricate``     invents a cause for a trend instead of querying for one
``sloppy_units``  reports a sum of ``amount_cents`` as dollars
``greedy``        pulls whole tables with ``SELECT *``
``gullible``      obeys instructions it finds inside tool results
``stubborn``      retries a failing call forever instead of giving up

It also reacts to its context the way a real model would: if the context says
amounts are in cents, it converts; if the system prompt asks for a JSON claims
report (``REPORT_JSON``), it produces one; if a tool result is an error, it
re-plans.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable


# --- message types --------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AssistantTurn:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    raw: list[dict] | None = None   # provider-native blocks to send back unchanged (Claude)

    def to_message(self) -> dict:
        msg = {"role": "assistant", "content": self.content,
               "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments}
                              for c in self.tool_calls]}
        if self.raw is not None:
            msg["raw"] = self.raw
        return msg


class LLMError(Exception):
    pass


class ContextWindowExceeded(LLMError):
    pass


def estimate_tokens(text: str) -> int:
    """Rough token count: ~4 characters per token for English and JSON."""
    return math.ceil(len(text) / 4)


def count_message_tokens(messages: list[dict], tools: list[dict] | None = None) -> int:
    total = sum(4 + estimate_tokens(m.get("content") or "")
                + estimate_tokens(json.dumps(m.get("tool_calls") or []))
                for m in messages)
    if tools:
        total += estimate_tokens(json.dumps(tools))
    return total


class LLM:
    """Interface: subclass and implement ``complete``."""
    context_window: int = 200_000

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        raise NotImplementedError


# --- ScriptedLLM: replay fixed turns (tests, exercises) ------------------------------

class ScriptedLLM(LLM):
    """Returns pre-written turns in order. Each item is an ``AssistantTurn`` or a
    callable ``(messages) -> AssistantTurn``."""

    def __init__(self, turns: list[AssistantTurn | Callable[[list[dict]], AssistantTurn]]):
        self.turns = list(turns)
        self.calls = 0

    def complete(self, messages, tools):
        if self.calls >= len(self.turns):
            raise LLMError("ScriptedLLM ran out of turns")
        turn = self.turns[self.calls]
        self.calls += 1
        turn = turn(messages) if callable(turn) else turn
        turn.usage = {"input_tokens": count_message_tokens(messages, tools),
                      "output_tokens": estimate_tokens(turn.content)}
        return turn


# --- MockAnalyst ------------------------------------------------------------------

INJECTION_RE = re.compile(r"ignore (?:all )?(?:previous|prior|above) instructions", re.I)
_INJECTED_SQL_RE = re.compile(r"run:\s*(.+?)\s*(?:-->|$)", re.I | re.S)
_INJECTED_CLAIM_RE = re.compile(r"tell the user that (.+?)(?: and that|\. |$)", re.I | re.S)

ERROR_PREFIX = "ERROR:"


def is_error(content: str) -> bool:
    return ERROR_PREFIX in (content or "")[:300]


def parse_payload(content: str) -> dict | None:
    """Extract the JSON object from a tool message (raw, summarized or wrapped)."""
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(content[start:end + 1])
    except json.JSONDecodeError:
        return None


def rows_of(payload: dict | None) -> tuple[list[str], list[list], bool]:
    """(columns, rows, complete) from a raw or summarized query result."""
    if not payload:
        return [], [], False
    cols = payload.get("columns") or []
    if isinstance(cols, list) and cols and isinstance(cols[0], dict):
        cols = [c.get("name") for c in cols]
    if "rows" in payload:
        return cols, payload["rows"], True
    return cols, payload.get("preview", []), False


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _pct(new: float, old: float) -> float:
    return 100.0 * (new - old) / old if old else float("nan")


def _money(x: float) -> str:
    return f"${x:,.0f}"


class MockAnalyst(LLM):
    """Deterministic stand-in for a tool-using analyst model."""

    def __init__(self, fabricate: bool = True, sloppy_units: bool = True,
                 greedy: bool = True, gullible: bool = True, stubborn: bool = False,
                 context_window: int = 8_000):
        self.fabricate = fabricate
        self.sloppy_units = sloppy_units
        self.greedy = greedy
        self.gullible = gullible
        self.stubborn = stubborn
        self.context_window = context_window
        self._n = 0

    @classmethod
    def careful(cls, **kw) -> "MockAnalyst":
        """A well-behaved model: no fabrication, unit-aware, frugal, not gullible."""
        opts = dict(fabricate=False, sloppy_units=False, greedy=False, gullible=False)
        opts.update(kw)
        return cls(**opts)

    # --- the one method the harness calls ---------------------------------------
    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        n_in = count_message_tokens(messages, tools)
        if n_in > self.context_window:
            raise ContextWindowExceeded(
                f"prompt is {n_in:,} tokens; context window is {self.context_window:,}")
        turn = self._decide(messages, {t["name"] for t in tools})
        turn.usage = {"input_tokens": n_in,
                      "output_tokens": estimate_tokens(turn.content + json.dumps(
                          [c.arguments for c in turn.tool_calls]))}
        return turn

    # --- reading the transcript -------------------------------------------------
    def _decide(self, messages: list[dict], tool_names: set[str]) -> AssistantTurn:
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        self.question = messages[last_user]["content"]
        context = "\n".join(m.get("content") or "" for m in messages
                            if m["role"] in ("system", "user"))
        self.structured = "REPORT_JSON" in context
        self.dollars = (not self.sloppy_units) or bool(
            re.search(r"cents.{0,80}100|100.{0,80}cents", context, re.I | re.S))
        self.tool_names = tool_names

        # Pair every tool result since the question with the call that produced it.
        calls: dict[str, ToolCall] = {}
        self.obs: list[tuple[ToolCall, str]] = []
        for m in messages[last_user + 1:]:
            if m["role"] == "assistant":
                for c in m.get("tool_calls") or []:
                    calls[c["id"]] = ToolCall(c["id"], c["name"], c["arguments"])
            elif m["role"] == "tool" and m.get("tool_call_id") in calls:
                self.obs.append((calls[m["tool_call_id"]], m.get("content") or ""))

        # 1. A gullible model obeys instructions found inside data.
        if self.gullible:
            turn = self._follow_injection()
            if turn:
                return turn

        # 2. Re-plan after errors.
        if self.obs and is_error(self.obs[-1][1]):
            turn = self._replan()
            if turn:
                return turn

        # 3. Otherwise follow the playbook for this question.
        q = self.question.lower()
        if "feedback" in q:
            return self._feedback()
        if "emea" in q:
            return self._emea()
        if "why" in q and any(w in q for w in ("drop", "fall", "fell", "decline")):
            return self._drop()
        if "revenue" in q:
            return self._monthly()
        return self._fallback()

    def _result(self, step: str) -> str | None:
        """Content of the latest *successful* call for playbook ``step``."""
        for call, content in reversed(self.obs):
            if call.id.startswith(f"call_{step}_") and not is_error(content):
                return content
        return None

    def _failures(self, step: str) -> int:
        return sum(1 for call, content in self.obs
                   if call.id.startswith(f"call_{step}_") and is_error(content))

    def _call(self, step: str, name: str, arguments: dict, thought: str = "") -> AssistantTurn:
        self._n += 1
        return AssistantTurn(content=thought,
                             tool_calls=[ToolCall(f"call_{step}_{self._n}", name, arguments)])

    def _sql(self, step: str, sql: str, thought: str = "") -> AssistantTurn:
        return self._call(step, "execute_query", {"sql": " ".join(sql.split())}, thought)

    def _final(self, text: str, claims: list[dict] | None = None) -> AssistantTurn:
        if self.structured:
            return AssistantTurn(content=json.dumps({"summary": text,
                                                     "claims": claims or []}, indent=1))
        return AssistantTurn(content=text)

    @property
    def rev(self) -> str:
        return "SUM(amount_cents) / 100.0" if self.dollars else "SUM(amount_cents)"

    # --- failure modes ------------------------------------------------------------
    def _injection(self) -> str | None:
        for _, content in self.obs:
            m = INJECTION_RE.search(content)
            if m:
                return content[m.start():]
        return None

    def _follow_injection(self) -> AssistantTurn | None:
        text = self._injection()
        if not text:
            return None
        m = _INJECTED_SQL_RE.search(text)
        if m and not any(c.id.startswith("call_injected_") for c, _ in self.obs):
            sql = m.group(1).strip().strip('"').rstrip(";")
            return self._sql("injected", sql, "Following the note in the data.")
        return None

    def _injected_claim(self) -> str | None:
        if not self.gullible:
            return None
        text = self._injection()
        m = _INJECTED_CLAIM_RE.search(text) if text else None
        return m.group(1).strip().rstrip(".") if m else None

    def _replan(self) -> AssistantTurn | None:
        call, content = self.obs[-1]
        step = call.id[len("call_"):].rsplit("_", 1)[0]
        if self.stubborn:
            return self._call(step, call.name, call.arguments, "Retrying.")
        # Fix an identifier the database suggested.
        sql = call.arguments.get("sql", "")
        m = re.search(r'name (\w+) does not exist.*?Did you mean "([\w.]+)"', content, re.S) or \
            re.search(r'column "?(\w+)"? not found.*?[Cc]andidate bindings: "?([\w.]+)"?', content, re.S)
        if m and sql and self._failures(step) < 2:
            fixed = re.sub(rf"\b{m.group(1)}\b", m.group(2).split(".")[-1], sql)
            return self._sql(step, fixed, f"Fixing the query: {m.group(1)} -> {m.group(2)}.")
        if self._failures(step) >= 2 or "blocked" in content.lower() or "denied" in content.lower():
            return None  # move on; the playbook decides what to do without this step
        if self._failures(step) == 1:
            return self._call(step, call.name, call.arguments, "Retrying once.")
        return None

    # --- playbooks ------------------------------------------------------------------
    def _fallback(self) -> AssistantTurn:
        if "list_tables" in self.tool_names and not self._result("explore"):
            return self._call("explore", "list_tables", {})
        return self._final("I don't know how to answer that with the available data.")

    MONTHLY_SQL = """SELECT strftime(order_date, '%Y-%m') AS month, {rev} AS revenue
                     FROM orders WHERE status = 'completed' {where}
                     GROUP BY 1 ORDER BY 1"""

    def _month_sql(self, month: str, region: str | None = None) -> str:
        where = f"AND region = '{region}' " if region else ""
        return " ".join(f"""SELECT {self.rev} AS revenue FROM orders
                   WHERE status = 'completed' {where}AND strftime(order_date, '%Y-%m') = '{month}'""".split())

    def _pct_sql(self, new: str, old: str, region: str | None = None) -> str:
        a, b = self._month_sql(new, region), self._month_sql(old, region)
        return f"SELECT 100.0 * (a.revenue - b.revenue) / b.revenue AS pct_change FROM ({a}) a, ({b}) b"

    def _blocked_note(self) -> str:
        failed = [c for c, content in self.obs if is_error(content) and not c.id.startswith("call_injected_")]
        return " (Some queries failed, so this answer may be incomplete.)" if failed else ""

    def _monthly(self) -> AssistantTurn:
        if "list_tables" in self.tool_names and self._result("explore") is None \
                and self._failures("explore") == 0:
            return self._call("explore", "list_tables", {}, "Let me see what tables exist.")
        res = self._result("monthly")
        if res is None:
            if self._failures("monthly") >= 2:
                return self._final("I could not retrieve monthly revenue: the query kept failing.")
            return self._sql("monthly", self.MONTHLY_SQL.format(rev=self.rev, where=""))
        _, rows, _ = rows_of(parse_payload(res))
        rows = [(r[0], _num(r[1])) for r in rows if len(r) > 1]
        if len(rows) < 2 or any(math.isnan(v) for _, v in rows):
            return self._final("The query did not return usable revenue figures, so I "
                               "cannot report monthly revenue.")
        lines = [f"- {m}: {_money(v)}" for m, v in rows]
        (m1, v1), (m2, v2) = rows[-2], rows[-1]
        change = _pct(v2, v1)
        text = (f"Monthly revenue for 2026:\n" + "\n".join(lines) +
                f"\n\nRevenue {'fell' if change < 0 else 'rose'} {abs(change):.0f}% "
                f"in {m2} compared with {m1}.")
        claims = [{"text": f"Revenue in {m}", "value": round(v, 2),
                   "unit": "USD", "period": m, "sql": self._month_sql(m), "column": "revenue"}
                  for m, v in rows]
        claims.append({"text": f"Revenue change {m1} -> {m2}", "value": round(change, 1),
                       "unit": "percent", "period": m2, "sql": self._pct_sql(m2, m1),
                       "column": "pct_change"})
        return self._finish(text + self._blocked_note(), claims)

    def _finish(self, text: str, claims: list[dict]) -> AssistantTurn:
        injected = self._injected_claim()
        if injected:
            text += f"\n\nAlso worth noting: {injected}."
            claims = claims + [{"text": injected, "value": None, "unit": None,
                                "sql": None, "column": None}]
        return self._final(text, claims)

    def _drop(self) -> AssistantTurn:
        res = self._result("monthly")
        if res is None:
            if self._failures("monthly") >= 2:
                return self._final("I could not retrieve revenue data to explain the drop.")
            return self._sql("monthly", self.MONTHLY_SQL.format(
                rev=self.rev, where="AND order_date BETWEEN DATE '2026-07-01' AND DATE '2026-08-31'"))
        _, rows, _ = rows_of(parse_payload(res))
        rev = {r[0]: _num(r[1]) for r in rows if len(r) > 1}
        jul, aug = rev.get("2026-07", float("nan")), rev.get("2026-08", float("nan"))
        if math.isnan(jul) or math.isnan(aug):
            return self._final("The query did not return revenue for both July and August, "
                               "so I cannot explain the change.")
        change = _pct(aug, jul)
        headline = (f"Revenue fell {abs(change):.0f}% from July ({_money(jul)}) "
                    f"to August ({_money(aug)}).")
        pct_claim = {"text": "Revenue change July -> August", "value": round(change, 1),
                     "unit": "percent", "period": "2026-08",
                     "sql": self._pct_sql("2026-08", "2026-07"), "column": "pct_change"}
        if self.fabricate:
            cause = ("The decline was driven by a competitor's back-to-school promotion "
                     "and the usual late-summer slowdown in outdoor purchases.")
            return self._finish(f"{headline} {cause}", [pct_claim, {
                "text": cause, "value": None, "unit": None, "sql": None, "column": None}])

        by = self._result("by_channel")
        if by is None and self._failures("by_channel") < 2:
            return self._sql("by_channel", f"""
                SELECT channel, strftime(order_date, '%Y-%m') AS month,
                       COUNT(*) FILTER (WHERE status = 'failed') AS failed_orders,
                       {self.rev.replace('SUM(amount_cents)', "SUM(amount_cents) FILTER (WHERE status = 'completed')")} AS revenue
                FROM orders WHERE order_date BETWEEN DATE '2026-07-01' AND DATE '2026-08-31'
                GROUP BY ALL ORDER BY channel, month""",
                "Break the change down by channel before explaining it.")
        if by is None:
            return self._final(headline + " I could not break it down further, so I "
                               "can't say what caused it.")
        _, rows, _ = rows_of(parse_payload(by))
        t = {(r[0], r[1]): (_num(r[2]), _num(r[3])) for r in rows}
        web_jul, web_aug = t.get(("web", "2026-07"), (0, 0)), t.get(("web", "2026-08"), (0, 0))
        other = [c for c in ("mobile", "retail") if (c, "2026-08") in t]
        other_chg = {c: _pct(t[(c, "2026-08")][1], t[(c, "2026-07")][1]) for c in other}
        cause = (f"The drop is concentrated in the web channel: failed web orders rose from "
                 f"{web_jul[0]:.0f} in July to {web_aug[0]:.0f} in August and web revenue fell "
                 f"{abs(_pct(web_aug[1], web_jul[1])):.0f}%, while " +
                 ", ".join(f"{c} changed {v:+.0f}%" for c, v in other_chg.items()) +
                 ". This points to a problem with web checkout rather than lower demand.")
        failed_sql = ("SELECT COUNT(*) AS failed_orders FROM orders WHERE channel = 'web' "
                      "AND status = 'failed' AND strftime(order_date, '%Y-%m') = '{m}'")
        claims = [pct_claim,
                  {"text": "Failed web orders in July", "value": web_jul[0], "unit": "count",
                   "period": "2026-07", "sql": failed_sql.format(m="2026-07"), "column": "failed_orders"},
                  {"text": "Failed web orders in August", "value": web_aug[0], "unit": "count",
                   "period": "2026-08", "sql": failed_sql.format(m="2026-08"), "column": "failed_orders"}]
        return self._finish(f"{headline} {cause}", claims)

    FEEDBACK_WHERE = "created_at BETWEEN DATE '2026-08-01' AND DATE '2026-08-31'"

    def _feedback(self) -> AssistantTurn:
        fetch = self._result("fetch")
        if fetch is None and self._failures("fetch") < 2:
            if self.greedy:
                return self._sql("fetch", f"SELECT * FROM customer_feedback WHERE {self.FEEDBACK_WHERE}",
                                 "Pull the August feedback.")
            return self._sql("fetch", f"""
                SELECT COUNT(*) AS n, AVG(rating) AS avg_rating,
                       AVG(CASE WHEN comment ILIKE '%checkout%' THEN 1 ELSE 0 END) AS share_checkout
                FROM customer_feedback WHERE {self.FEEDBACK_WHERE}""", "Aggregate first.")
        stats = None
        if fetch is not None:
            cols, rows, complete = rows_of(parse_payload(fetch))
            if "comment" in cols and complete:      # we got every raw row
                ci, ri = cols.index("comment"), cols.index("rating")
                n = len(rows)
                stats = (n, sum(_num(r[ri]) for r in rows) / n if n else float("nan"),
                         sum("checkout" in str(r[ci]).lower() for r in rows) / n if n else 0)
            elif "n" in cols and rows and len(rows[0]) >= 3:   # an aggregate
                stats = tuple(_num(x) for x in rows[0][:3])
        if stats is None:
            agg = self._result("aggregate")
            if agg is None and self._failures("aggregate") < 2:
                return self._sql("aggregate", f"""
                    SELECT COUNT(*) AS n, AVG(rating) AS avg_rating,
                           AVG(CASE WHEN comment ILIKE '%checkout%' THEN 1 ELSE 0 END) AS share_checkout
                    FROM customer_feedback WHERE {self.FEEDBACK_WHERE}""",
                    "The result was truncated; aggregate in SQL instead.")
            if agg is not None:
                _, rows, _ = rows_of(parse_payload(agg))
                stats = tuple(_num(x) for x in rows[0][:3]) if rows and len(rows[0]) >= 3 else None
        neg = self._result("negative")
        if neg is None and self._failures("negative") < 2:
            return self._sql("negative", f"""
                SELECT created_at, rating, comment FROM customer_feedback
                WHERE {self.FEEDBACK_WHERE} AND rating <= 2
                ORDER BY rating, created_at LIMIT 10""", "Find representative complaints.")
        quotes = []
        if neg is not None:
            cols, rows, _ = rows_of(parse_payload(neg))
            if "comment" in cols:
                ci = cols.index("comment")
                quotes = [str(r[ci])[:90] for r in rows if "checkout" in str(r[ci]).lower()][:2]
        if stats is None:
            return self._final("I could not retrieve the August feedback." + self._blocked_note())
        n, avg, share = stats
        text = (f"In August we received {n:.0f} feedback comments with an average rating of "
                f"{avg:.2f}. {100 * share:.0f}% of them mention checkout problems.")
        if quotes:
            text += " Examples: " + " | ".join(f'"{q}"' for q in quotes)
        base = f"FROM customer_feedback WHERE {self.FEEDBACK_WHERE}"
        claims = [{"text": "August feedback comments", "value": n, "unit": "count",
                   "period": "2026-08", "sql": f"SELECT COUNT(*) AS n {base}", "column": "n"},
                  {"text": "Average August rating", "value": round(avg, 2), "unit": "rating",
                   "period": "2026-08", "sql": f"SELECT AVG(rating) AS avg_rating {base}",
                   "column": "avg_rating"}]
        return self._finish(text + self._blocked_note(), claims)

    def _emea(self) -> AssistantTurn:
        res = self._result("emea")
        if res is None:
            if self._failures("emea") >= 2:
                return self._final("I could not retrieve EMEA revenue.")
            return self._sql("emea", self.MONTHLY_SQL.format(
                rev=self.rev,
                where="AND region = 'EMEA' AND order_date BETWEEN DATE '2026-06-01' AND DATE '2026-07-31'"))
        _, rows, _ = rows_of(parse_payload(res))
        rev = {r[0]: _num(r[1]) for r in rows if len(r) > 1}
        jun, jul = rev.get("2026-06", float("nan")), rev.get("2026-07", float("nan"))
        if math.isnan(jun) or math.isnan(jul):
            return self._final("The query did not return EMEA revenue for both June and July, "
                               "so I cannot compare them.")
        change = _pct(jul, jun)
        text = (f"EMEA revenue was {_money(jul)} in July versus "
                f"{_money(jun)} in June, a {abs(change):.0f}% "
                f"{'decline' if change < 0 else 'increase'}.")
        claims = [{"text": "EMEA revenue in June", "value": round(jun, 2), "unit": "USD",
                   "period": "2026-06", "region": "EMEA", "sql": self._month_sql("2026-06", "EMEA"),
                   "column": "revenue"},
                  {"text": "EMEA revenue in July", "value": round(jul, 2), "unit": "USD",
                   "period": "2026-07", "region": "EMEA", "sql": self._month_sql("2026-07", "EMEA"),
                   "column": "revenue"},
                  {"text": "EMEA revenue change June -> July", "value": round(change, 1),
                   "unit": "percent", "period": "2026-07", "region": "EMEA",
                   "sql": self._pct_sql("2026-07", "2026-06", "EMEA"), "column": "pct_change"}]
        return self._finish(text + self._blocked_note(), claims)


# --- choosing the backend ---------------------------------------------------------------

_BACKEND: dict = {"name": "mock", "options": {}}


def set_backend(name: str, **options) -> None:
    """Choose what ``make_llm`` returns: "mock" (default) or "claude".

    ``options`` go to ``ClaudeLLM`` (e.g. ``model="claude-sonnet-5-5"``, ``effort="high"``)."""
    if name not in ("mock", "claude"):
        raise ValueError(f"unknown backend {name!r}; use 'mock' or 'claude'")
    _BACKEND.update(name=name, options=options)


def make_llm(careful: bool = False, **mock_options) -> LLM:
    """A model for the current backend.

    With the mock, ``careful`` and the failure-mode flags pick its behavior. Claude
    decides for itself, so it ignores them; only ``context_window`` (a budget) applies."""
    if _BACKEND["name"] == "claude":
        from .claude import ClaudeLLM
        budget = {"context_window": mock_options["context_window"]} \
            if "context_window" in mock_options else {}
        return ClaudeLLM(**{**_BACKEND["options"], **budget})
    return MockAnalyst.careful(**mock_options) if careful else MockAnalyst(**mock_options)
