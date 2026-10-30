"""Claude as the analyst's model, through the Anthropic Messages API.

``ClaudeLLM`` implements the same ``complete(messages, tools)`` interface as the
mock, so the harness doesn't change. It translates the harness's message format
to Claude's:

harness                                   Claude Messages API
{"role": "system"}                        top-level ``system``
{"role": "assistant", "tool_calls"}       ``text`` + ``tool_use`` blocks
{"role": "tool", "tool_call_id"}          ``tool_result`` blocks in a user turn

Claude's own response blocks (including thinking) are kept on the assistant
message as ``raw`` and sent back unchanged on the next call. The API checks
that a conversation's earlier turns are unchanged before it accepts thinking
blocks; when the harness edits history (``ContextCompactor`` does), the adapter
drops all thinking blocks, which is always allowed.

Credentials come from the environment (``ANTHROPIC_API_KEY``) as usual.
"""

from __future__ import annotations

import anthropic

from .llm import (ERROR_PREFIX, LLM, AssistantTurn, ContextWindowExceeded, LLMError,
                  ToolCall, count_message_tokens)

DEFAULT_MODEL = "claude-opus-5-5"
# Models that accept server-side refusal fallbacks with the "default" routing
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
THINKING_BLOCKS = ("thinking", "redacted_thinking")


def to_claude(messages: list[dict]) -> tuple[str, list[dict]]:
    """Harness messages -> (system prompt, Claude messages)."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    out: list[dict] = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            content = m.get("raw") or (
                ([{"type": "text", "text": m["content"]}] if m.get("content") else [])
                + [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]}
                   for c in m.get("tool_calls") or []])
            if content:
                out.append({"role": "assistant", "content": content})
        elif m["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"],
                     "content": m["content"] or "(empty result)"}
            if (m["content"] or "").startswith(ERROR_PREFIX):
                block["is_error"] = True
            prev = out[-1] if out else None
            if prev and prev["role"] == "user" and isinstance(prev["content"], list) \
                    and prev["content"][-1].get("type") == "tool_result":
                prev["content"].append(block)  # all results for one turn go in one message
            else:
                out.append({"role": "user", "content": [block]})
    return system, out


def strip_thinking(messages: list[dict]) -> list[dict]:
    return [{**m, "content": [b for b in m["content"] if b.get("type") not in THINKING_BLOCKS]}
            if m["role"] == "assistant" and isinstance(m["content"], list) else m
            for m in messages]


def has_thinking(messages: list[dict]) -> bool:
    return any(isinstance(m["content"], list) and
               any(b.get("type") in THINKING_BLOCKS for b in m["content"])
               for m in messages if m["role"] == "assistant")


class ClaudeLLM(LLM):
    """Claude behind the harness's ``LLM`` interface.

    ``context_window`` is a client-side budget: prompts estimated above it raise
    ``ContextWindowExceeded`` before anything is sent, exactly like the mock."""

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "medium",
                 max_tokens: int = 16_000, context_window: int = 1_000_000,
                 fallbacks: bool | None = None, client: anthropic.Anthropic | None = None):
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.fallbacks = model in FALLBACK_MODELS if fallbacks is None else fallbacks
        self.thinking_dropped = 0      # how many requests went out without thinking blocks
        self._last: tuple | None = None

    def _history_edited(self, system: str, tools: list[dict], msgs: list[dict]) -> bool:
        """Same conversation as the last call, but not an append-only extension of it?"""
        if self._last is None or not msgs:
            return False
        last_system, last_tools, last_msgs = self._last
        if not last_msgs or last_msgs[0] != msgs[0]:
            return False                              # a new conversation
        return (system, tools) != (last_system, last_tools) or msgs[:len(last_msgs)] != last_msgs

    def _create(self, system: str, msgs: list[dict], tools: list[dict]):
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, messages=msgs,
                      output_config={"effort": self.effort},
                      cache_control={"type": "ephemeral"})
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if self.fallbacks:  # on a refusal, the API re-runs the request on a fallback model
            return self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        return self.client.messages.create(**kwargs)

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        n_in = count_message_tokens(messages, tools)
        if n_in > self.context_window:
            raise ContextWindowExceeded(
                f"prompt is ~{n_in:,} tokens; context budget is {self.context_window:,}")
        system, msgs = to_claude(messages)
        edited = self._history_edited(system, tools, msgs)
        self._last = (system, tools, msgs)
        if edited and has_thinking(msgs):
            msgs = strip_thinking(msgs)
            self.thinking_dropped += 1
        try:
            try:
                resp = self._create(system, msgs, tools)
            except anthropic.BadRequestError:
                if not has_thinking(msgs):
                    raise
                # Some other history edit invalidated the thinking blocks: drop them all and retry.
                resp = self._create(system, strip_thinking(msgs), tools)
                self.thinking_dropped += 1
        except anthropic.APIError as e:
            raise LLMError(f"Claude API error: {e}") from e

        if resp.stop_reason == "refusal":
            category = getattr(resp.stop_details, "category", None)
            raise LLMError(f"Claude declined the request (category: {category})")
        calls = [ToolCall(b.id, b.name, b.input) for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason == "max_tokens" and calls:
            raise LLMError(f"response hit max_tokens={self.max_tokens} in the middle of a tool call")
        u = resp.usage
        usage = {"input_tokens": u.input_tokens + (u.cache_read_input_tokens or 0)
                 + (u.cache_creation_input_tokens or 0),
                 "output_tokens": u.output_tokens}
        return AssistantTurn(
            content="\n".join(b.text for b in resp.content if b.type == "text"),
            tool_calls=calls, usage=usage,
            raw=[b.to_dict(exclude_none=True) for b in resp.content])
