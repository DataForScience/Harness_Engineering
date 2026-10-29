"""Verifying AI reports: every number must be re-derivable.

The model must return a JSON report whose claims each carry the SQL that
produced them. The ``Verifier`` then, for every claim:

1. **evidence**  -- refuses claims with no query behind them (fabrications)
2. **value**     -- re-runs the query and compares the number
3. **units**     -- catches cents reported as dollars
4. **coverage**  -- flags periods whose data is incomplete or missing

Only after that does an LLM judge look at the report, and only to grade
clarity and usefulness. Judges are bad at arithmetic and great at prose.
"""

from __future__ import annotations

import calendar
import json
import math
from dataclasses import dataclass, field
from typing import Callable

from sqlglot import exp

from .guardrails import inspect_sql
from .mcp_client import MCPClient
from .permissions import classify_sql


# --- the claims contract -------------------------------------------------------------------

REPORT_PROMPT = """REPORT_JSON: return your final answer as a JSON object:
{"summary": "<plain-language answer>",
 "claims": [{"text": "...", "value": <number>, "unit": "USD|percent|count|rating",
             "period": "YYYY-MM", "region": "<optional>", "sql": "<query that returns the value>",
             "column": "<result column holding the value>"}]}
Every number in the summary must appear as a claim with its SQL."""

SQLRunner = Callable[[str], tuple[list[str], list[list]]]


def make_sql_runner(client: MCPClient, allowed_tables: set[str]) -> SQLRunner:
    """Run verification queries through the same SQL guard as the agent."""
    def run(sql: str):
        d = inspect_sql(sql, allowed_tables)
        if d.action != "allow":
            raise PermissionError(d.reason)
        r = client.call_tool("execute_query", {"sql": sql})
        if not r.ok or not r.data:
            raise RuntimeError(r.text[:300])
        return r.data["columns"], r.data["rows"]
    return run


def parse_report(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0:
        raise ValueError("no JSON report found")
    report = json.loads(text[start:end + 1])
    report.setdefault("claims", [])
    return report


# --- checks and their outcome --------------------------------------------------------------

@dataclass
class Check:
    claim: str
    status: str            # verified | unsupported | mismatch | unit_error | incomplete_period | error
    detail: str = ""
    expected: float | None = None
    actual: float | None = None


@dataclass
class Verification:
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.status == "verified" for c in self.checks)

    def table(self):
        import pandas as pd
        return pd.DataFrame([c.__dict__ for c in self.checks])


# --- units -----------------------------------------------------------------------------------

def converts_cents(sql: str, column: str) -> bool:
    """Does the expression behind ``column`` divide an ``*_cents`` column by 100?"""
    info = classify_sql(sql)
    for tree in info.trees:
        for proj in tree.find_all(exp.Alias):
            if proj.alias != column:
                continue
            cents = [c for c in proj.find_all(exp.Column) if c.name.endswith("_cents")]
            if not cents:
                return True        # not a cents column at all
            return any(isinstance(d, exp.Div) and d.expression.this in ("100", "100.0")
                       for d in proj.find_all(exp.Div))
    return True                     # can't tell; let the value check decide


# --- the verifier ----------------------------------------------------------------------------

class Verifier:
    def __init__(self, run_sql: SQLRunner, rel_tol: float = 0.005):
        self.run_sql = run_sql
        self.rel_tol = rel_tol

    def _value(self, sql: str, column: str | None) -> float:
        cols, rows = self.run_sql(sql)
        if not rows:
            raise ValueError("query returned no rows")
        idx = cols.index(column) if column in cols else 0
        return float(rows[0][idx])

    def _close(self, a: float, b: float) -> bool:
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=0.051)

    def coverage(self, period: str, region: str | None = None) -> tuple[int, int, int]:
        """(days with orders, days expected, failed loads) for a YYYY-MM period."""
        year, month = map(int, period.split("-"))
        expected = calendar.monthrange(year, month)[1]
        where = f"strftime(order_date, '%Y-%m') = '{period}'"
        where += f" AND region = '{region}'" if region else ""
        _, rows = self.run_sql(f"SELECT COUNT(DISTINCT order_date) AS days, MAX(order_date) AS last_day "
                               f"FROM orders WHERE {where}")
        days = int(rows[0][0])
        log_where = f"strftime(load_date, '%Y-%m') = '{period}' AND status = 'failed'"
        log_where += f" AND region = '{region}'" if region else ""
        _, rows = self.run_sql(f"SELECT COUNT(*) AS failed FROM ingestion_log WHERE {log_where}")
        return days, expected, int(rows[0][0])

    def check(self, claim: dict) -> Check:
        text = claim.get("text", "")
        sql, column, value = claim.get("sql"), claim.get("column"), claim.get("value")
        if not sql:
            return Check(text, "unsupported", "no query backs this claim")
        if value is None:
            return Check(text, "unsupported", "claim has no numeric value to check")
        try:
            actual = self._value(sql, column)
        except Exception as e:  # noqa: BLE001
            return Check(text, "error", f"could not re-run: {e}", value)
        if not self._close(value, actual):
            for factor in (100, 0.01):
                if self._close(value, actual * factor):
                    return Check(text, "unit_error", f"off by a factor of {factor}", value, actual)
            return Check(text, "mismatch", "re-run value differs", value, actual)
        if claim.get("unit") == "USD" and not converts_cents(sql, column):
            return Check(text, "unit_error", "USD claim sums an *_cents column without /100",
                         value, actual)
        if claim.get("period") and "orders" in classify_sql(sql).tables:
            days, expected, failed = self.coverage(claim["period"], claim.get("region"))
            if days < expected or failed:
                where = f" for {claim['region']}" if claim.get("region") else ""
                return Check(text, "incomplete_period",
                             f"{claim['period']}{where}: data for {days}/{expected} days, "
                             f"{failed} failed loads", value, actual)
        return Check(text, "verified", "", value, actual)

    def verify(self, report: dict) -> Verification:
        return Verification([self.check(c) for c in report.get("claims", [])])


# --- wiring it in: an on_final hook --------------------------------------------------------

class VerifiedAnswer:
    """``on_final`` hook: verify the report and render the result for the user."""

    def __init__(self, verifier: Verifier):
        self.verifier = verifier
        self.last: Verification | None = None
        self.report: dict | None = None

    def __call__(self, text: str) -> str:
        try:
            self.report = parse_report(text)
        except ValueError as e:
            self.last = Verification([Check("report", "error", str(e))])
            return f"UNVERIFIED (could not parse report: {e})\n\n{text}"
        self.last = self.verifier.verify(self.report)
        failed = [c for c in self.last.checks if c.status != "verified"]
        lines = [self.report.get("summary", "")]
        if not self.last.checks:
            lines.append("\nVerification: the report makes no checkable claims.")
        elif failed:
            lines.append(f"\nVerification: {len(failed)} of {len(self.last.checks)} claims "
                         "did not pass:")
            lines += [f"  - [{c.status}] {c.claim}: {c.detail}" for c in failed]
        else:
            lines.append(f"\nVerification: all {len(self.last.checks)} claims re-derived.")
        return "\n".join(lines)


# --- LLM-as-judge, for clarity and usefulness only --------------------------------------------

JUDGE_PROMPT = """You are reviewing a data report for a business reader.
The numbers have ALREADY been verified by re-running queries; do not assess correctness.
Score 1-5 for:
- clarity: is it easy to read, with the answer first and no unexplained jargon?
- usefulness: does it answer the question, quantify the effect, and note caveats?
Return JSON: {"clarity": n, "usefulness": n, "comments": "..."}"""


class MockJudge:
    """Deterministic stand-in for an LLM judge using the rubric above."""

    JARGON = ("strftime", "sum(", "amount_cents", "group by", "null", "sql")

    def score(self, question: str, summary: str, verification: Verification | None = None) -> dict:
        sentences = [s for s in summary.replace("\n", " ").split(". ") if s.strip()]
        avg_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        clarity = 5 - (avg_len > 25) - (avg_len > 40) - any(j in summary.lower() for j in self.JARGON)
        has_numbers = any(ch.isdigit() for ch in summary)
        caveat = any(w in summary.lower() for w in ("incomplete", "missing", "partial", "caveat", "only"))
        on_topic = len(set(question.lower().split()) & set(summary.lower().split())) >= 2
        usefulness = 1 + has_numbers + caveat + on_topic + (verification is None or verification.passed)
        comments = []
        if not caveat:
            comments.append("no caveats about data quality")
        if avg_len > 25:
            comments.append("long sentences")
        return {"clarity": max(clarity, 1), "usefulness": usefulness,
                "comments": "; ".join(comments) or "clear and useful"}
