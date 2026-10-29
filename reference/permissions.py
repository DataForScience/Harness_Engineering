"""Permissioned tools: decide in code, before the server sees a call.

Every tool call passes through a ``PermissionGate``:

1. **Allowlist** -- only tools the policy names exist for the model, and calls to
   anything else are refused (models do hallucinate tool names).
2. **Argument validation** -- arguments must match the tool's JSON Schema.
3. **Rules** -- small functions that look at a call and return a ``Decision``
   (allow / ask / deny). The first rule with an opinion wins.
4. **Approval** -- "ask" goes to an approver (a human, a ticket, a second
   system). With no approver, "ask" means "no".

Every decision is written to an audit log.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import sqlglot
from sqlglot import exp

from .agent import ToolBox
from .llm import ToolCall

logging.getLogger("sqlglot").setLevel(logging.ERROR)


@dataclass
class Decision:
    action: str          # "allow" | "ask" | "deny"
    reason: str = ""
    rule: str = ""


Rule = Callable[[ToolCall], "Decision | None"]


# --- argument validation ----------------------------------------------------------

_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
          "object": dict, "array": list, "null": type(None)}


def validate_args(schema: dict, args: dict) -> list[str]:
    """Check ``args`` against the subset of JSON Schema that tool specs use."""
    if not isinstance(args, dict):
        return ["arguments must be an object"]
    errors = []
    props = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in args:
            errors.append(f"missing required argument {name!r}")
    for name, value in args.items():
        if name not in props:
            if schema.get("additionalProperties", True) is False:
                errors.append(f"unexpected argument {name!r}")
            continue
        spec = props[name]
        options = spec.get("anyOf", [spec])
        types = [o.get("type") for o in options if "type" in o]
        if types and not any(isinstance(value, _TYPES[t]) and
                             not (t in ("integer", "number") and isinstance(value, bool))
                             for t in types):
            errors.append(f"{name!r} must be {' or '.join(types)}, got {type(value).__name__}")
        if isinstance(value, str) and "maxLength" in spec and len(value) > spec["maxLength"]:
            errors.append(f"{name!r} is longer than {spec['maxLength']} characters")
    return errors


# --- SQL classification -----------------------------------------------------------

READ_STATEMENTS = (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Describe, exp.Show)


@dataclass
class SQLInfo:
    statements: list[str] = field(default_factory=list)
    tables: set[str] = field(default_factory=set)
    columns: set[str] = field(default_factory=set)
    read_only: bool = False
    error: str | None = None
    trees: list = field(default_factory=list)


def classify_sql(sql: str) -> SQLInfo:
    """Parse ``sql`` (DuckDB dialect) and describe what it would do."""
    info = SQLInfo()
    try:
        trees = [t for t in sqlglot.parse(sql, read="duckdb") if t is not None]
    except sqlglot.errors.ParseError as e:
        info.error = f"could not parse SQL: {str(e).splitlines()[0]}"
        return info
    info.trees = trees
    info.statements = [type(t).__name__ for t in trees]
    for t in trees:
        ctes = {c.alias_or_name for c in t.find_all(exp.CTE)}
        for table in t.find_all(exp.Table):
            if table.name and table.name not in ctes:
                info.tables.add(table.name)
        info.columns |= {c.name for c in t.find_all(exp.Column)}
    info.read_only = bool(trees) and all(isinstance(t, READ_STATEMENTS) for t in trees)
    return info


# --- rules ------------------------------------------------------------------------

def sql_read_only(call: ToolCall) -> Decision | None:
    """Only single, read-only statements reach the database."""
    if "sql" not in call.arguments:
        return None
    info = classify_sql(call.arguments["sql"])
    if info.error:
        return Decision("deny", info.error, "sql_read_only")
    if len(info.statements) != 1:
        return Decision("deny", f"exactly one statement allowed, got {len(info.statements)}",
                        "sql_read_only")
    if not info.read_only:
        return Decision("deny", f"{info.statements[0]} statements are not allowed (read-only)",
                        "sql_read_only")
    return None


def sensitive_tables(tables: set[str]) -> Rule:
    """Ask for approval before touching any of ``tables``."""
    def rule(call: ToolCall) -> Decision | None:
        if "sql" not in call.arguments:
            return None
        touched = classify_sql(call.arguments["sql"]).tables & tables
        if touched:
            return Decision("ask", f"query reads sensitive table(s) {sorted(touched)}",
                            "sensitive_tables")
        return None
    rule.__name__ = "sensitive_tables"
    return rule


# --- the gate -----------------------------------------------------------------------

class PermissionGate:
    """A ``before_tool`` hook that enforces a tool policy.

    ``policy`` maps tool name -> "allow" | "ask" | "deny"; tools not listed are denied.
    ``rules`` run in order after validation; the first non-None Decision wins.
    ``approver(call, decision) -> bool`` handles "ask"."""

    def __init__(self, toolbox: ToolBox, policy: dict[str, str], rules: list[Rule] = (),
                 approver: Callable[[ToolCall, Decision], bool] | None = None):
        self.toolbox = toolbox
        self.policy = policy
        self.rules = list(rules)
        self.approver = approver
        self.audit: list[dict] = []

    def visible(self) -> ToolBox:
        """Hide every tool the policy doesn't allow or ask for."""
        for name in list(self.toolbox.tools):
            if self.policy.get(name, "deny") == "deny":
                self.toolbox.remove(name)
        return self.toolbox

    def decide(self, call: ToolCall) -> Decision:
        level = self.policy.get(call.name, "deny")
        if level == "deny" or call.name not in self.toolbox.tools:
            return Decision("deny", f"tool {call.name!r} is not permitted", "allowlist")
        errors = validate_args(self.toolbox.tools[call.name].input_schema, call.arguments)
        if errors:
            return Decision("deny", "; ".join(errors), "schema")
        for rule in self.rules:
            d = rule(call)
            if d is not None:
                return d
        return Decision(level, "tool policy", "policy")

    def __call__(self, call: ToolCall) -> str | None:
        d = self.decide(call)
        approved = None
        if d.action == "ask":
            approved = bool(self.approver and self.approver(call, d))
        self.audit.append({"time": time.strftime("%H:%M:%S"), "tool": call.name,
                           "args": call.arguments, "action": d.action, "rule": d.rule,
                           "reason": d.reason, "approved": approved})
        if d.action == "allow" or approved:
            return None
        if d.action == "ask":
            return f"needs human approval ({d.reason}) and was not approved"
        return f"denied by rule '{d.rule}': {d.reason}"
