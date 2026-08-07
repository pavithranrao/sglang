import asyncio
import unittest
from unittest.mock import Mock, patch

from utils import (
    collect_stream_events,
    event_payloads,
    event_types,
    find_completed_event,
    make_serving,
)

from sglang.srt.entrypoints.openai.protocol import (
    RequestResponseMetadata,
    ResponsesRequest,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _StreamFixture:
    def __init__(self, serving, request, *, require_reasoning=False):
        self.serving = serving
        self.request = request
        self.require_reasoning = require_reasoning
        self.request_metadata = RequestResponseMetadata(request_id=request.request_id)

    def run(self, chunks):
        async def gen():
            for ch in chunks:
                yield ch

        async def collect():
            return await collect_stream_events(
                self.serving.responses_stream_generator_non_harmony(
                    self.request,
                    sampling_params={},
                    result_generator=gen(),
                    model_name="x",
                    tokenizer=Mock(),
                    request_metadata=self.request_metadata,
                    require_reasoning=self.require_reasoning,
                )
            )

        return asyncio.run(collect())


def _engine_chunk(
    text,
    completion_tokens,
    *,
    finish=False,
    token_logprobs=None,
    top_logprobs=None,
):
    meta = {
        "id": "rid",
        "prompt_tokens": 5,
        "completion_tokens": completion_tokens,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "finish_reason": {"type": "stop"} if finish else None,
    }
    if token_logprobs is not None:
        meta["output_token_logprobs"] = token_logprobs
    if top_logprobs is not None:
        meta["output_top_logprobs"] = top_logprobs
    return {"text": text, "meta_info": meta}


class NonHarmonyStreamTestCase(unittest.TestCase):
    def test_reasoning_parser_uses_processed_reasoning_state(self):
        serving = make_serving()
        serving.reasoning_parser = "deepseek-r1"
        request = ResponsesRequest(model="x", input="hi", stream=True, store=False)

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.ReasoningParser"
        ) as parser_cls:
            parser_cls.return_value.parse_stream_chunk.return_value = (None, "done")
            fixture = _StreamFixture(serving, request, require_reasoning=True)
            fixture.run([_engine_chunk("done", 1, finish=True)])

        self.assertTrue(parser_cls.call_args.kwargs["force_reasoning"])

    def test_emits_typed_sse_events_in_order(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None

        request = ResponsesRequest(model="x", input="hi", stream=True, store=False)
        fixture = _StreamFixture(serving, request)
        events = fixture.run(
            [
                _engine_chunk("Hel", 1),
                _engine_chunk("Hello", 2),
                _engine_chunk("Hello world", 4, finish=True),
            ]
        )

        types = event_types(events)
        self.assertEqual(types[0], "response.created")
        self.assertEqual(types[1], "response.in_progress")
        for ev in (
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
        ):
            self.assertIn(ev, types)
        self.assertEqual(types[-1], "response.completed")

        seqs = [p["sequence_number"] for p in event_payloads(events)]
        self.assertEqual(seqs, list(range(len(seqs))))

    def test_required_tool_choice_emits_function_call_events(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None

        request = ResponsesRequest(
            model="x",
            input="hi",
            stream=True,
            store=False,
            tool_choice="required",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }
            ],
        )
        payload = '[{"name": "get_weather", "parameters": {"city": "Beijing"}}]'

        chunks = []
        sent = 0
        while sent < len(payload):
            sent += min(8, len(payload) - sent)
            chunks.append(
                _engine_chunk(payload[:sent], sent, finish=sent == len(payload))
            )

        fixture = _StreamFixture(serving, request)
        events = fixture.run(chunks)
        types = event_types(events)

        self.assertIn("response.function_call_arguments.delta", types)
        self.assertIn("response.function_call_arguments.done", types)
        self.assertIn("response.output_item.added", types)
        self.assertIn("response.output_item.done", types)
        self.assertNotIn("response.output_text.delta", types)

        added_kinds = [
            payload["item"]["type"]
            for payload in event_payloads(events)
            if payload.get("type") == "response.output_item.added"
        ]
        self.assertIn("function_call", added_kinds)

    def test_final_output_preserves_text_tool_text_order(self):
        from sglang.srt.function_call.core_types import (
            StreamingParseResult,
            ToolCallItem,
        )

        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = "qwen3_coder"

        request = ResponsesRequest(
            model="x",
            input="hi",
            stream=True,
            store=False,
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }
            ],
        )

        scripted = [
            StreamingParseResult(normal_text="I'll check.", calls=[]),
            StreamingParseResult(
                normal_text="",
                calls=[
                    ToolCallItem(
                        tool_index=0,
                        name="get_weather",
                        parameters='{"city": "Beijing"}',
                    )
                ],
            ),
            StreamingParseResult(normal_text="It's sunny.", calls=[]),
        ]
        chunks = [
            _engine_chunk(" " * 3, 3),
            _engine_chunk(" " * 10, 10),
            _engine_chunk(" " * 14, 14, finish=True),
        ]

        script_iter = iter(scripted)

        def fake_parse_stream_chunk(delta):
            sp = next(script_iter)
            return sp.normal_text, sp.calls

        with patch(
            "sglang.srt.entrypoints.openai.serving_responses.FunctionCallParser"
        ) as parser_cls:
            parser_cls.return_value.detector.supports_structural_tag.return_value = True
            parser_cls.return_value.parse_stream_chunk.side_effect = (
                fake_parse_stream_chunk
            )
            fixture = _StreamFixture(serving, request)
            events = fixture.run(chunks)

        completed = find_completed_event(events)
        output = completed["response"]["output"]
        kinds = [item["type"] for item in output]
        self.assertEqual(kinds, ["message", "function_call", "message"])
        self.assertEqual(output[0]["content"][0]["text"], "I'll check.")
        self.assertEqual(output[1]["name"], "get_weather")
        self.assertEqual(output[2]["content"][0]["text"], "It's sunny.")


class StreamLogprobsTestCase(unittest.TestCase):
    def _delta_events(self, events):
        payloads = event_payloads(events)
        return [p for p in payloads if p["type"] == "response.output_text.delta"]

    def _done_payload(self, events):
        for p in event_payloads(events):
            if p["type"] == "response.output_text.done":
                return p
        raise AssertionError("response.output_text.done missing")

    def test_delta_events_carry_per_chunk_logprobs(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None

        request = ResponsesRequest(
            model="x", input="hi", stream=True, top_logprobs=2, store=False
        )

        # Each chunk carries cumulative text + cumulative logprobs.
        chunk1_lp = [(-0.5, 313, "Hel")]
        chunk1_top = [[(-0.5, 313, "Hel"), (-1.2, 999, "Hi")]]
        chunk2_lp = [(-0.5, 313, "Hel"), (-0.3, 414, "lo")]
        chunk2_top = [
            [(-0.5, 313, "Hel"), (-1.2, 999, "Hi")],
            [(-0.3, 414, "lo"), (-2.0, 888, "Lo")],
        ]

        chunks = [
            _engine_chunk(
                "Hel", 1, token_logprobs=chunk1_lp, top_logprobs=chunk1_top
            ),
            _engine_chunk(
                "Hello", 2, token_logprobs=chunk2_lp, top_logprobs=chunk2_top
            ),
            _engine_chunk("Hello world", 3, finish=True),
        ]

        fixture = _StreamFixture(serving, request)
        events = fixture.run(chunks)
        deltas = self._delta_events(events)

        # Two delta events ("Hel" then "lo"), each with logprobs.
        self.assertEqual(len(deltas), 2)
        self.assertEqual(len(deltas[0]["logprobs"]), 1)
        self.assertEqual(deltas[0]["logprobs"][0]["token"], "Hel")
        self.assertEqual(deltas[0]["logprobs"][0]["logprob"], -0.5)
        self.assertEqual(len(deltas[0]["logprobs"][0]["top_logprobs"]), 2)

        # Second delta only carries the new token (n_prev_logprobs slicing).
        self.assertEqual(len(deltas[1]["logprobs"]), 1)
        self.assertEqual(deltas[1]["logprobs"][0]["token"], "lo")

    def test_done_event_carries_full_accumulated_logprobs(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None

        request = ResponsesRequest(
            model="x", input="hi", stream=True, top_logprobs=1, store=False
        )

        chunk1_lp = [(-0.5, 313, "Hel")]
        chunk1_top = [[(-0.5, 313, "Hel")]]
        chunk2_lp = [(-0.5, 313, "Hel"), (-0.3, 414, "lo")]

        chunks = [
            _engine_chunk(
                "Hel", 1, token_logprobs=chunk1_lp, top_logprobs=chunk1_top
            ),
            _engine_chunk(
                "Hello", 2, token_logprobs=chunk2_lp, top_logprobs=None
            ),
            _engine_chunk("Hello world", 3, finish=True),
        ]

        fixture = _StreamFixture(serving, request)
        events = fixture.run(chunks)
        done = self._done_payload(events)

        # The done event should hold all accumulated logprobs.
        self.assertEqual(len(done["logprobs"]), 2)
        self.assertEqual(done["logprobs"][0]["token"], "Hel")
        self.assertEqual(done["logprobs"][1]["token"], "lo")

    def test_no_logprobs_when_not_requested(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None

        request = ResponsesRequest(
            model="x", input="hi", stream=True, store=False
        )

        chunks = [
            _engine_chunk(
                "Hel", 1, token_logprobs=[(-0.5, 313, "Hel")]
            ),
            _engine_chunk("Hello", 2, finish=True),
        ]

        fixture = _StreamFixture(serving, request)
        events = fixture.run(chunks)
        deltas = self._delta_events(events)

        self.assertTrue(deltas)
        for d in deltas:
            self.assertEqual(d["logprobs"], [])

    def test_include_list_triggers_streaming_logprobs(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None

        request = ResponsesRequest(
            model="x",
            input="hi",
            stream=True,
            top_logprobs=0,
            include=["message.output_text.logprobs"],
            store=False,
        )

        chunks = [
            _engine_chunk(
                "Hi", 1, token_logprobs=[(-0.1, 42, "Hi")]
            ),
            _engine_chunk("Hi!", 2, finish=True),
        ]

        fixture = _StreamFixture(serving, request)
        events = fixture.run(chunks)
        deltas = self._delta_events(events)

        self.assertEqual(len(deltas), 1)
        self.assertEqual(len(deltas[0]["logprobs"]), 1)
        self.assertEqual(deltas[0]["logprobs"][0]["token"], "Hi")
        # top_logprobs=0 -> empty alternatives.
        self.assertEqual(len(deltas[0]["logprobs"][0]["top_logprobs"]), 0)


if __name__ == "__main__":
    unittest.main()
