# Solution: establish the ground truth by querying the MCP server yourself.
from harness.exercises import check_01
from harness.mcp_client import MCPClient

with MCPClient() as client:
    def query(sql):
        result = client.call_tool("execute_query", {"sql": sql})
        return result.data["rows"]

    # (a) Revenue is completed orders only, and amount_cents is in cents.
    august = query("""SELECT SUM(amount_cents) / 100.0 AS revenue FROM orders
                      WHERE status = 'completed'
                        AND order_date BETWEEN DATE '2026-08-01' AND DATE '2026-08-31'""")
    # (b) Compare channels between July and August: failed orders explode on the web.
    by_channel = query("""SELECT channel,
                                 COUNT(*) FILTER (WHERE status = 'failed' AND month(order_date) = 7) AS failed_jul,
                                 COUNT(*) FILTER (WHERE status = 'failed' AND month(order_date) = 8) AS failed_aug
                          FROM orders GROUP BY channel ORDER BY failed_aug DESC""")
    # (c) Count the days that actually have EMEA orders in July.
    emea_days = query("""SELECT COUNT(DISTINCT order_date) AS days FROM orders
                         WHERE region = 'EMEA'
                           AND order_date BETWEEN DATE '2026-07-01' AND DATE '2026-07-31'""")

answers = {
    "august_revenue_usd": float(august[0][0]),
    "drop_channel": by_channel[0][0],
    "emea_july_days": int(emea_days[0][0]),
}
print(answers)
check_01(answers)
