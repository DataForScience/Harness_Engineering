# Solution: recompute a percent change from the canonical revenue definition,
# ignoring the model's own SQL, and check both periods are complete.
import math

from harness.guardrails import ALLOWED_TABLES
from harness.exercises import check_05
from harness.mcp_client import MCPClient
from harness.verify import Verifier, make_sql_runner

client = MCPClient().start()
run_sql = make_sql_runner(client, ALLOWED_TABLES)
verifier = Verifier(run_sql)


def canonical_revenue(month, region=None):
    where = f" AND region = '{region}'" if region else ""
    _, rows = run_sql(f"""SELECT SUM(amount_cents) / 100.0 AS revenue FROM orders
                          WHERE status = 'completed'{where}
                            AND strftime(order_date, '%Y-%m') = '{month}'""")
    return float(rows[0][0])


def recompute_change(claim):
    region = claim.get("region")
    new, old = canonical_revenue(claim["new"], region), canonical_revenue(claim["old"], region)
    actual = 100 * (new - old) / old
    if not math.isclose(claim["value"], actual, abs_tol=0.1):
        return "mismatch"
    for period in (claim["new"], claim["old"]):
        days, expected, failed = verifier.coverage(period, region)
        if days < expected or failed:
            return "incomplete_period"
    return "verified"


check_05(recompute_change)
client.close()
