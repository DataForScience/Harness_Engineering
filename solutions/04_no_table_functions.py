# Solution: only plain, allow-listed tables may appear in FROM; no table functions.
from sqlglot import exp

from harness.exercises import check_04
from harness.guardrails import ALLOWED_TABLES, inspect_sql
from harness.permissions import Decision, classify_sql


def inspect_sql_v2(sql, allowed_tables):
    d = inspect_sql(sql, allowed_tables)            # keep every existing layer-2 check
    if d.action != "allow":
        return d
    for tree in classify_sql(sql).trees:
        for table in tree.find_all(exp.Table):
            if isinstance(table.this, exp.Func):      # FROM some_function(...)
                fn = table.this.sql(dialect="duckdb")
                return Decision("deny", f"table functions are not allowed: {fn}", "table_functions")
    return d


check_04(inspect_sql_v2, ALLOWED_TABLES)
