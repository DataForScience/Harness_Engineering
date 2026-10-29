# Solution: ask for approval only when a query exposes PII columns.
from sqlglot import exp

from harness.exercises import check_02
from harness.permissions import Decision, classify_sql

PII_COLUMNS = {"name", "email"}


def pii_columns(call):
    if call.name != "execute_query" or "sql" not in call.arguments:
        return None                                   # not our business
    info = classify_sql(call.arguments["sql"])
    if "customers" not in info.tables:
        return None
    # SELECT * / c.* expose every column; the * inside COUNT(*) does not.
    stars = any(isinstance(p, exp.Star) or (isinstance(p, exp.Column) and isinstance(p.this, exp.Star))
                for t in info.trees for sel in t.find_all(exp.Select) for p in sel.expressions)
    exposed = info.columns & PII_COLUMNS
    if exposed or stars:
        what = sorted(exposed) if exposed else "* (all columns)"
        return Decision("ask", f"query exposes PII: {what}", "pii_columns")
    return None                                       # aggregates over customers are fine


check_02(pii_columns)
