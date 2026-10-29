# Solution: a circuit breaker that fails fast after repeated failures.
import time

from harness.exercises import check_06
from harness.mcp_client import ToolResult


class CircuitBreaker:
    def __init__(self, invoke, threshold=3, cooldown=5.0):
        self.invoke = invoke
        self.threshold = threshold
        self.cooldown = cooldown
        self.failures = 0
        self.opened_at = None                       # None means closed

    def __call__(self, name, arguments):
        if self.opened_at is not None:
            if time.monotonic() - self.opened_at < self.cooldown:
                return ToolResult(False, f"circuit open: {name} failed {self.failures} times in a row; "
                                         "not calling it until the cooldown ends")
            # cooldown over: half-open, let one probe call through
        try:
            result = self.invoke(name, arguments)
        except Exception as e:  # noqa: BLE001
            result = ToolResult(False, f"{type(e).__name__}: {e}")
        if result.ok:
            self.failures, self.opened_at = 0, None
        else:
            self.failures += 1
            if self.failures >= self.threshold:
                self.opened_at = time.monotonic()
        return result


check_06(CircuitBreaker)
