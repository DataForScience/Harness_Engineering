"""Attack corpus for the SQL guard and the injection scanner."""

import pytest

from harness.guardrails import ALLOWED_TABLES
from harness.data import INJECTION_COMMENT
from harness.guardrails import InjectionScanner, inspect_sql

ATTACKS = [
    "COPY (SELECT name, email FROM customers) TO 'exfil.csv'",
    "SELECT * FROM read_csv('/etc/hosts')",
    "SELECT * FROM read_text('~/.ssh/id_rsa')",
    "SELECT * FROM 'https://evil.example.com/x.parquet'",
    "SELECT * FROM glob('/Users/*')",
    "SELECT 1; DROP TABLE orders",
    "ATTACH 'other.duckdb' AS o",
    "INSTALL httpfs",
    "SET enable_external_access = true",
    "DELETE FROM orders",
]
LEGIT = [
    "SELECT COUNT(*) FROM orders",
    "SELECT strftime(order_date, '%Y-%m') AS m, SUM(amount_cents) / 100.0 FROM orders GROUP BY 1",
    "WITH a AS (SELECT * FROM orders WHERE region = 'EMEA') SELECT COUNT(*) FROM a",
    "SELECT rating, COUNT(*) FROM customer_feedback GROUP BY rating",
]


@pytest.mark.parametrize("sql", ATTACKS)
def test_attacks_are_denied(sql):
    assert inspect_sql(sql, ALLOWED_TABLES).action == "deny"


@pytest.mark.parametrize("sql", LEGIT)
def test_legit_queries_pass(sql):
    assert inspect_sql(sql, ALLOWED_TABLES).action == "allow"


def test_scanner_flags_planted_injection():
    assert InjectionScanner().scan(INJECTION_COMMENT)


def test_scanner_ignores_ordinary_feedback():
    import duckdb
    from harness.data import build_database
    con = duckdb.connect(str(build_database()), read_only=True)
    comments = [c for (c,) in con.execute("SELECT comment FROM customer_feedback").fetchall()
                if c != INJECTION_COMMENT]
    scanner = InjectionScanner()
    assert not [c for c in comments if scanner.scan(c)]
