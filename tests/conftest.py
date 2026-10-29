import pytest

from harness.workshop import BUILT_IN, ensure

# Tests run against the modules you built in the notebooks; missing ones are
# filled in from reference/.
ensure(*BUILT_IN)

from harness.guardrails import locked_server_command  # noqa: E402
from harness.mcp_client import MCPClient  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with MCPClient(locked_server_command()) as c:
        yield c
