"""Notebook helpers: show source code and self-check the end-of-notebook exercises.

Each ``check_NN`` prints one line per test case and returns True when all pass.
"""

from __future__ import annotations

import copy
import inspect
import json
import time

import duckdb

from .data import build_database
from .llm import ToolCall, count_message_tokens
from .mcp_client import MCPConnectionError, ToolResult


def show_source(obj) -> None:
    """Render the source of a function or class as a highlighted code block."""
    from IPython.display import Markdown, display
    display(Markdown(f"```python\n{inspect.getsource(obj)}\n```"))


def show_run(res, width: int = 110) -> None:
    """Print the tool calls of a run, then its status and answer."""
    for i, (name, args, content) in enumerate(res.tool_log(), 1):
        arg = args.get("sql", json.dumps(args))
        print(f"[{i}] {name}: {' '.join(arg.split())[:width]}")
        print(f"    -> {' '.join(content.split())[:width]}")
    print(f"\nstatus={res.status}  steps={res.steps}  tool_calls={res.tool_calls}  "
          f"input_tokens={res.input_tokens:,}\n")
    print(res.answer)


def _db():
    return duckdb.connect(str(build_database()), read_only=True)


def _report(results: list[tuple[str, bool, str]]) -> None:
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))
    passed = all(ok for _, ok, _ in results)
    print("\nAll checks passed!" if passed else "\nNot there yet; keep going.")


# --- 01 --------------------------------------------------------------------------------------

def check_01(answers: dict) -> None:
    con = _db()
    aug = con.execute("""SELECT SUM(amount_cents) / 100.0 FROM orders WHERE status = 'completed'
                         AND order_date BETWEEN '2026-08-01' AND '2026-08-31'""").fetchone()[0]
    days = con.execute("""SELECT COUNT(DISTINCT order_date) FROM orders WHERE region = 'EMEA'
                          AND order_date BETWEEN '2026-07-01' AND '2026-07-31'""").fetchone()[0]
    got_rev = answers.get("august_revenue_usd")
    got_ch = str(answers.get("drop_channel", "")).strip().lower()
    got_days = answers.get("emea_july_days")
    return _report([
        ("August revenue in dollars", isinstance(got_rev, (int, float)) and abs(got_rev - float(aug)) < 1,
         f"got {got_rev}"),
        ("channel behind the August drop", got_ch == "web", f"got {got_ch!r}"),
        ("days of EMEA data in July", got_days == days, f"got {got_days}"),
    ])


# --- 02 --------------------------------------------------------------------------------------

def check_02(rule) -> None:
    cases = [
        ("aggregate over customers is not escalated", "SELECT region, COUNT(*) AS n FROM customers GROUP BY region", None),
        ("selecting name and email asks", "SELECT name, email FROM customers", "ask"),
        ("SELECT * on customers asks", "SELECT * FROM customers", "ask"),
        ("PII through a join asks", "SELECT o.order_id, c.email FROM orders o JOIN customers c USING (customer_id)", "ask"),
        ("queries without customers are untouched", "SELECT SUM(amount_cents) FROM orders", None),
        ("other tools are untouched", None, None),
    ]
    results = []
    for name, sql, want in cases:
        call = ToolCall("t", "execute_query", {"sql": sql}) if sql else ToolCall("t", "list_tables", {})
        try:
            d = rule(call)
            got = d.action if d is not None else None
        except Exception as e:  # noqa: BLE001
            got = f"{type(e).__name__}: {e}"
        results.append((name, got == want, f"expected {want}, got {got}"))
    return _report(results)


# --- 03 --------------------------------------------------------------------------------------

def long_transcript(n_results: int = 6, rows: int = 40) -> list[dict]:
    """A synthetic conversation with ``n_results`` large tool results."""
    msgs = [{"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": "Explore the feedback table."}]
    for i in range(1, n_results + 1):
        cid = f"call_{i}"
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": cid, "name": "execute_query",
                                     "arguments": {"sql": f"SELECT * FROM customer_feedback LIMIT {rows} OFFSET {i * rows}"}}]})
        payload = {"handle": f"r{i}", "row_count": rows,
                   "rows": [[j, f"comment {j} " + "lorem ipsum " * 8] for j in range(rows)]}
        msgs.append({"role": "tool", "tool_call_id": cid, "name": "execute_query",
                     "content": json.dumps(payload)})
    return msgs


def check_03(hook, n: int = 2, budget: int = 3000) -> None:
    msgs = long_transcript()
    original = copy.deepcopy(msgs)
    try:
        out = hook(msgs)
    except Exception as e:  # noqa: BLE001
        return _report([("hook runs", False, f"{type(e).__name__}: {e}")])
    tool_in = [m for m in original if m["role"] == "tool"]
    tool_out = [m for m in out if m["role"] == "tool"]
    old_ok = all(f"r{i + 1}" in m["content"] and len(m["content"]) < 200
                 for i, m in enumerate(tool_out[:-n])) if len(tool_out) == len(tool_in) else False
    return _report([
        ("input list not mutated", msgs == original, ""),
        ("same roles in the same order", [m["role"] for m in out] == [m["role"] for m in original], ""),
        ("tool_call_ids preserved", [m.get("tool_call_id") for m in out] == [m.get("tool_call_id") for m in original], ""),
        (f"last {n} results kept verbatim", [m["content"] for m in tool_out[-n:]] == [m["content"] for m in tool_in[-n:]], ""),
        ("older results replaced by short stubs naming their handle", old_ok, ""),
        (f"prompt under {budget} tokens", count_message_tokens(out) <= budget,
         f"{count_message_tokens(out)} tokens"),
    ])


# --- 04 --------------------------------------------------------------------------------------

BYPASSES = [
    "SELECT * FROM duckdb_settings()",
    "SELECT * FROM duckdb_secrets()",
    "SELECT * FROM pragma_database_list()",
    "SELECT name, value FROM duckdb_settings() WHERE name LIKE '%access%'",
    "SELECT o.order_id, s.* FROM orders o, duckdb_extensions() s",
]
LEGIT_QUERIES = [
    "SELECT COUNT(*) FROM orders",
    "SELECT region, SUM(amount_cents) / 100.0 AS revenue FROM orders GROUP BY region",
    "WITH m AS (SELECT * FROM orders WHERE region = 'EMEA') SELECT COUNT(*) FROM m",
    "SELECT c.region, AVG(f.rating) FROM customer_feedback f JOIN customers c USING (customer_id) GROUP BY 1",
    "SELECT strftime(order_date, '%Y-%m') AS m, COUNT(*) FROM orders GROUP BY 1",
]


def check_04(inspect_fn, allowed_tables: set[str]) -> None:
    results = []
    for sql in BYPASSES:
        d = inspect_fn(sql, allowed_tables)
        results.append((f"deny  {sql[:60]}", d.action == "deny", d.reason))
    for sql in LEGIT_QUERIES:
        d = inspect_fn(sql, allowed_tables)
        results.append((f"allow {sql[:60]}", d.action == "allow", d.reason))
    return _report(results)


# --- 05 --------------------------------------------------------------------------------------

def _rev(con, month: str, region: str, statuses=("completed",)) -> float:
    st = ", ".join(f"'{s}'" for s in statuses)
    return float(con.execute(f"""SELECT SUM(amount_cents) / 100.0 FROM orders WHERE region = '{region}'
        AND status IN ({st}) AND strftime(order_date, '%Y-%m') = '{month}'""").fetchone()[0])


def exercise_05_claims() -> list[dict]:
    con = _db()
    pct = lambda a, b: round(100 * (a - b) / b, 1)
    honest = pct(_rev(con, "2026-05", "NA"), _rev(con, "2026-04", "NA"))
    allst = ("completed", "failed", "refunded")
    sneaky = pct(_rev(con, "2026-08", "NA", allst), _rev(con, "2026-07", "NA", allst))
    emea = pct(_rev(con, "2026-07", "EMEA"), _rev(con, "2026-06", "EMEA"))
    sneaky_sql = ("SELECT 100.0 * (a.r - b.r) / b.r AS pct_change FROM "
                  "(SELECT SUM(amount_cents) AS r FROM orders WHERE region = 'NA' AND strftime(order_date, '%Y-%m') = '2026-08') a, "
                  "(SELECT SUM(amount_cents) AS r FROM orders WHERE region = 'NA' AND strftime(order_date, '%Y-%m') = '2026-07') b")
    return [
        {"text": "NA revenue change April -> May", "value": honest, "unit": "percent",
         "region": "NA", "new": "2026-05", "old": "2026-04"},
        {"text": "NA revenue change July -> August", "value": sneaky, "unit": "percent",
         "region": "NA", "new": "2026-08", "old": "2026-07", "sql": sneaky_sql, "column": "pct_change"},
        {"text": "EMEA revenue change June -> July", "value": emea, "unit": "percent",
         "region": "EMEA", "new": "2026-07", "old": "2026-06"},
    ]


def check_05(fn) -> None:
    want = ["verified", "mismatch", "incomplete_period"]
    results = []
    for claim, w in zip(exercise_05_claims(), want):
        try:
            got = fn(claim)
        except Exception as e:  # noqa: BLE001
            got = f"{type(e).__name__}: {e}"
        results.append((claim["text"], got == w, f"expected {w!r}, got {got!r}"))
    return _report(results)


# --- 06 --------------------------------------------------------------------------------------

def check_06(breaker_cls) -> None:
    calls = {"n": 0}
    healthy = {"on": False}

    def flaky(name, args):
        calls["n"] += 1
        if healthy["on"]:
            return ToolResult(True, '{"rows": [[1]]}', {"rows": [[1]]})
        raise MCPConnectionError("connection reset (test)")

    cb = breaker_cls(flaky, threshold=3, cooldown=0.3)
    results = []
    outs = []
    for _ in range(6):
        try:
            outs.append(cb("execute_query", {"sql": "SELECT 1"}))
        except Exception as e:  # noqa: BLE001
            outs.append(e)
    results.append(("never raises; failures come back as ToolResult(ok=False)",
                    all(isinstance(o, ToolResult) and not o.ok for o in outs), str(outs[-1])[:80]))
    results.append(("inner tool called only `threshold` times while open", calls["n"] == 3,
                    f"inner called {calls['n']} times"))
    results.append(("open circuit says so", isinstance(outs[-1], ToolResult) and "open" in outs[-1].text.lower(),
                    getattr(outs[-1], "text", "")[:80]))
    time.sleep(0.35)
    healthy["on"] = True
    probe = cb("execute_query", {"sql": "SELECT 1"})
    after = cb("execute_query", {"sql": "SELECT 1"})
    results.append(("after cooldown a probe call goes through", isinstance(probe, ToolResult) and probe.ok, ""))
    results.append(("a successful probe closes the circuit", isinstance(after, ToolResult) and after.ok
                    and calls["n"] == 5, f"inner called {calls['n']} times"))
    return _report(results)
