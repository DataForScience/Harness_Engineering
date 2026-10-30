"""Offline tests for the Claude adapter: a fake client stands in for the API."""

from types import SimpleNamespace as NS

import anthropic
import httpx2 as httpx
import pytest

from harness.claude import ClaudeLLM, has_thinking, to_claude
from harness.llm import LLMError, make_llm, set_backend, MockAnalyst

TOOLS = [{"name": "execute_query", "description": "Run SQL",
          "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}}}}]


def block(type, **kw):
    return NS(type=type, **kw, to_dict=lambda exclude_none=True: {"type": type, **kw})


def response(*blocks, stop="end_turn"):
    return NS(content=list(blocks), stop_reason=stop, stop_details=None,
              usage=NS(input_tokens=10, cache_read_input_tokens=5,
                       cache_creation_input_tokens=0, output_tokens=3))


class FakeClient:
    def __init__(self, *responses, fail_first_with_thinking=False):
        self.responses = list(responses)
        self.requests = []
        self.fail = fail_first_with_thinking
        self.messages = NS(create=self._create)
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        thinking = any(isinstance(m["content"], list) and
                       any(b.get("type") == "thinking" for b in m["content"])
                       for m in kw["messages"] if m["role"] == "assistant")
        if self.fail and thinking:
            self.fail = False
            raise anthropic.BadRequestError(
                "thinking block invalid", body=None,
                response=httpx.Response(400, request=httpx.Request("POST", "https://x")))
        return self.responses.pop(0)


def test_message_translation_groups_tool_results():
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "name": "execute_query", "arguments": {"sql": "1"}},
                {"id": "b", "name": "execute_query", "arguments": {"sql": "2"}}]},
            {"role": "tool", "tool_call_id": "a", "name": "execute_query", "content": "{}"},
            {"role": "tool", "tool_call_id": "b", "name": "execute_query", "content": "ERROR: boom"}]
    system, out = to_claude(msgs)
    assert system == "sys"
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    results = out[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert results[1]["is_error"] and "is_error" not in results[0]


def test_turn_parsing_and_raw_replay():
    client = FakeClient(
        response(block("thinking", thinking="", signature="s1"),
                 block("tool_use", id="t1", name="execute_query", input={"sql": "SELECT 1"}),
                 stop="tool_use"),
        response(block("text", text="done")))
    llm = ClaudeLLM(client=client)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    turn = llm.complete(msgs, TOOLS)
    assert turn.tool_calls[0].arguments == {"sql": "SELECT 1"}
    assert turn.usage == {"input_tokens": 15, "output_tokens": 3}
    msgs += [turn.to_message(), {"role": "tool", "tool_call_id": "t1", "name": "execute_query",
                                 "content": "{}"}]
    assert llm.complete(msgs, TOOLS).content == "done"
    sent = client.requests[1]["messages"][1]["content"]
    assert sent[0]["type"] == "thinking", "thinking blocks go back unchanged"
    assert client.requests[1]["fallbacks"] == "default"


def test_history_edit_drops_thinking():
    thinking_turn = lambda i: response(block("thinking", thinking="", signature=f"s{i}"),
                                       block("tool_use", id=f"t{i}", name="execute_query",
                                             input={"sql": "x"}), stop="tool_use")
    client = FakeClient(thinking_turn(1), thinking_turn(2), response(block("text", text="ok")))
    llm = ClaudeLLM(client=client)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    for i in (1, 2):
        msgs += [llm.complete(msgs, TOOLS).to_message(),
                 {"role": "tool", "tool_call_id": f"t{i}", "name": "execute_query", "content": "big"}]
    assert has_thinking(client.requests[1]["messages"]), "append-only: thinking kept"
    msgs[3] = {**msgs[3], "content": "[elided]"}       # what ContextCompactor does
    llm.complete(msgs, TOOLS)
    assert not has_thinking(client.requests[2]["messages"]), "edited history: thinking dropped"
    assert llm.thinking_dropped == 1


def test_bad_request_with_thinking_retries_without_it():
    client = FakeClient(
        response(block("thinking", thinking="", signature="s"),
                 block("tool_use", id="t1", name="execute_query", input={}), stop="tool_use"),
        response(block("text", text="ok")), fail_first_with_thinking=True)
    llm = ClaudeLLM(client=client)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    msgs += [llm.complete(msgs, TOOLS).to_message(),
             {"role": "tool", "tool_call_id": "t1", "name": "execute_query", "content": "r"}]
    assert llm.complete(msgs, TOOLS).content == "ok"
    assert len(client.requests) == 3 and llm.thinking_dropped == 1


def test_refusal_becomes_llm_error():
    client = FakeClient(response(block("text", text=""), stop="refusal"))
    with pytest.raises(LLMError, match="declined"):
        ClaudeLLM(client=client).complete([{"role": "user", "content": "q"}], TOOLS)


def test_backend_switch():
    set_backend("claude", model="claude-sonnet-5-5")
    try:
        llm = make_llm(careful=True, fabricate=True, context_window=8000)
        assert isinstance(llm, ClaudeLLM)
        assert llm.model == "claude-sonnet-5-5" and llm.context_window == 8000
    finally:
        set_backend("mock")
    assert isinstance(make_llm(), MockAnalyst)
