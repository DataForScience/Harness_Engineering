"""A minimal, synchronous MCP client over stdio (no SDK, no framework).

MCP over stdio is newline-delimited JSON-RPC 2.0: we write one JSON object per
line to the server's stdin and read one per line from its stdout. A background
thread drains stdout into a queue so every request can have a timeout.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .data import DB_PATH, build_database

PROTOCOL_VERSION = "2025-06-18"


class MCPError(Exception):
    """The server answered with a JSON-RPC error."""


class MCPTimeout(MCPError, TimeoutError):
    """No response within the deadline."""


class MCPConnectionError(MCPError, ConnectionError):
    """The server process is gone or its pipe is broken."""


@dataclass
class ToolResult:
    """What a tool call produced, normalized.

    ``ok`` is False when the server flagged the call with ``isError``.
    ``data`` is the parsed JSON payload when the server returned JSON text.
    """
    ok: bool
    text: str
    data: dict | None = None
    meta: dict = field(default_factory=dict)


def parse_tool_result(result: dict) -> ToolResult:
    text = "\n".join(c.get("text", "") for c in result.get("content", [])
                     if c.get("type") == "text")
    data = None
    start = text.find("{")
    if start >= 0:
        try:
            data = json.loads(text[start:])
        except json.JSONDecodeError:
            data = None
    return ToolResult(ok=not result.get("isError", False), text=text, data=data)


def server_command(db_path: str | Path = DB_PATH, *extra: str) -> list[str]:
    """Command line for the DuckDB MCP server (``mcp-server-motherduck``)."""
    bindir = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")
    exe = shutil.which("mcp-server-motherduck", path=str(bindir)) or \
        shutil.which("mcp-server-motherduck")
    if exe is None:
        raise FileNotFoundError("mcp-server-motherduck not found; run `uv sync`")
    return [exe, "--db-path", str(db_path), *extra]


class MCPClient:
    """Spawn an MCP server and talk to it.

    >>> with MCPClient() as client:
    ...     tools = client.list_tools()
    ...     result = client.call_tool("execute_query", {"sql": "SELECT 1"})
    """

    def __init__(self, command: list[str] | None = None, timeout: float = 30.0):
        if command is None:
            build_database()
            command = server_command()
        self.command = command
        self.timeout = timeout
        self.proc: subprocess.Popen | None = None
        self.server_info: dict = {}
        self._next_id = 0
        self._responses: queue.Queue = queue.Queue()
        self._lock = threading.Lock()

    # --- lifecycle -------------------------------------------------------
    def start(self) -> "MCPClient":
        self.proc = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._responses = queue.Queue()
        threading.Thread(target=self._reader, args=(self.proc, self._responses),
                         daemon=True).start()
        init = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "d4sci-harness", "version": "0.1"},
        }, timeout=60)
        self.server_info = init.get("serverInfo", {})
        self.notify("notifications/initialized")
        return self

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def restart(self) -> None:
        self.close()
        self.start()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def __enter__(self) -> "MCPClient":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # --- JSON-RPC ------------------------------------------------------------
    @staticmethod
    def _reader(proc: subprocess.Popen, out: queue.Queue) -> None:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                out.put(json.loads(line))
            except json.JSONDecodeError:
                continue  # servers sometimes log to stdout; ignore non-JSON
        out.put(None)  # EOF sentinel: the server exited

    def _send(self, message: dict) -> None:
        if not self.alive:
            raise MCPConnectionError("MCP server is not running")
        try:
            self.proc.stdin.write(json.dumps(message) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise MCPConnectionError(f"pipe to MCP server broken: {e}") from e

    def notify(self, method: str, params: dict | None = None) -> None:
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)

    def request(self, method: str, params: dict | None = None,
                timeout: float | None = None) -> dict:
        with self._lock:
            self._next_id += 1
            rid = self._next_id
            msg = {"jsonrpc": "2.0", "id": rid, "method": method}
            if params is not None:
                msg["params"] = params
            self._send(msg)
            deadline = time.monotonic() + (timeout or self.timeout)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MCPTimeout(f"{method} timed out after {timeout or self.timeout}s")
                try:
                    resp = self._responses.get(timeout=remaining)
                except queue.Empty:
                    continue
                if resp is None:
                    raise MCPConnectionError("MCP server exited")
                if resp.get("id") != rid:
                    continue  # notification or a late reply to a timed-out request
                if "error" in resp:
                    raise MCPError(resp["error"].get("message", str(resp["error"])))
                return resp["result"]

    # --- MCP methods -------------------------------------------------------
    def list_tools(self) -> list[dict]:
        return self.request("tools/list", {})["tools"]

    def call_tool(self, name: str, arguments: dict[str, Any],
                  timeout: float | None = None) -> ToolResult:
        result = self.request("tools/call", {"name": name, "arguments": arguments},
                              timeout=timeout)
        return parse_tool_result(result)
