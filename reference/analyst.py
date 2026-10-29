"""The complete harness, assembled: one function that wires every component."""

from __future__ import annotations

from dataclasses import dataclass

from .agent import DEFAULT_SYSTEM_PROMPT, Agent, ToolBox
from .context import ContextCompactor, MemoryStore, ResultCompactor, ResultStore
from .guardrails import ALLOWED_TABLES, SPOTLIGHT_PROMPT, InjectionScanner, sql_guard, spotlighter
from .llm import LLM
from .mcp_client import MCPClient
from .permissions import PermissionGate, sensitive_tables, sql_read_only
from .supervisor import Budget, FaultInjector, ReplanAdvisor, ResilientInvoker, Tracer
from .verify import REPORT_PROMPT, VerifiedAnswer, Verifier, make_sql_runner

POLICY = {"execute_query": "allow", "list_tables": "allow", "list_columns": "allow",
          "expand_result": "allow"}


@dataclass
class Harness:
    agent: Agent
    gate: PermissionGate
    scanner: InjectionScanner
    store: ResultStore
    verifier: VerifiedAnswer
    tracer: Tracer
    budget: Budget
    faults: FaultInjector | None

    def run(self, question: str):
        self.budget.reset()
        return self.agent.run(question)


def build_analyst(llm: LLM, client: MCPClient, memory: MemoryStore | None = None,
                  question_hint: str = "", fault_plan: dict[int, str] | None = None,
                  fault_rate: float = 0.0, seed: int = 0, approver=None,
                  budget: Budget | None = None, timeout: float = 10.0) -> Harness:
    tools = ToolBox(client)
    store = ResultStore()
    store.register_tool(tools)
    gate = PermissionGate(tools, POLICY, rules=[sql_read_only, sql_guard(ALLOWED_TABLES),
                                                sensitive_tables({"customers"})],
                          approver=approver)
    gate.visible()
    scanner = InjectionScanner()
    tracer = Tracer()
    budget = budget or Budget()
    faults = FaultInjector(tools.call, client, plan=fault_plan, rate=fault_rate, seed=seed) \
        if (fault_plan or fault_rate) else None
    invoker = ResilientInvoker(client, call=faults or tools.call, timeout=timeout,
                               tracer=tracer, seed=seed)
    system = "\n\n".join([DEFAULT_SYSTEM_PROMPT, SPOTLIGHT_PROMPT, REPORT_PROMPT])
    if memory is not None:
        system = memory.system_prompt(system, question_hint)
    verifier = VerifiedAnswer(Verifier(make_sql_runner(client, ALLOWED_TABLES)))
    agent = Agent(llm, tools, system_prompt=system, max_steps=12,
                  before_tool=[gate],
                  after_tool=[scanner, ResultCompactor(store), ReplanAdvisor(), spotlighter],
                  prepare_context=ContextCompactor(budget=6000),
                  on_final=verifier, invoke=invoker, budget=budget, tracer=tracer)
    return Harness(agent, gate, scanner, store, verifier, tracer, budget, faults)
