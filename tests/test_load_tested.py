"""Unit tests for the load-testing helpers and metric aggregation."""

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

# Running this file directly puts tests/, rather than the repository root, on
# sys.path. Add the root so the project's src namespace package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import load_tested


class FakeResponse:
    def __init__(self, status=200, lines=()):
        self.status = status
        self.content = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class FakeContent:
    def __init__(self, lines):
        self.lines = lines

    def __aiter__(self):
        self._iterator = iter(self.lines)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            raise StopAsyncIteration


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs))
        return self.response


def sse_line(data):
    return f"data: {json.dumps(data)}\n".encode()


def test_load_prompts_jsonl_reads_prompt_and_text_and_skips_bad_lines(tmp_path, capsys):
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text(
        '{"prompt": "first"}\n'
        '{"text": "second"}\n'
        '\n'
        'not-json\n'
        '{"text": "   "}\n',
        encoding="utf-8",
    )

    assert load_tested.load_prompts_jsonl(dataset, 2) == ["first", "second"]
    assert "skipping malformed JSONL line 4" in capsys.readouterr().out


def test_load_prompts_jsonl_rejects_too_few_usable_prompts(tmp_path):
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text('{"text": "only one"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Requires 2 lines.*found only 1 usable prompts"):
        load_tested.load_prompts_jsonl(dataset, 2)


def test_send_stream_request_parses_usage_and_builds_payload(monkeypatch):
    lines = [
        b": keepalive\n",
        b"data: not-json\n",
        sse_line({"choices": [{"delta": {"content": "hello"}}]}),
        sse_line({"choices": [{"delta": {"content": " world"}}]}),
        sse_line({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2}}),
        b"data: [DONE]\n",
    ]
    session = FakeSession(FakeResponse(lines=FakeContent(lines)))
    counter = Counter()
    times = iter([10.0, 10.1, 10.3, 10.4])
    monkeypatch.setattr(load_tested.time, "time", lambda: next(times))

    result = asyncio.run(
        load_tested.send_stream_request(
            session,
            "http://localhost/v1/chat/completions",
            "test-model",
            "question",
            counter,
            system_prompt="be concise",
            response_format={"type": "json_object"},
            max_tokens=8,
            ignore_eos=True,
        )
    )

    endpoint, request = session.calls[0]
    payload = request["json"]
    assert endpoint == "http://localhost/v1/chat/completions"
    assert payload["messages"] == [
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "question"},
    ]
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["ignore_eos"] is True
    assert payload["max_tokens"] == 8
    assert result["ttft"] == pytest.approx(0.1)
    assert result["e2e"] == pytest.approx(0.4)
    assert result["tpot"] == pytest.approx(0.2)
    assert result["itls"] == pytest.approx([0.2])
    assert result["output_tokens"] == 2
    assert result["input_tokens"] == 7
    assert not counter


def test_send_stream_request_records_http_error():
    session = FakeSession(FakeResponse(status=503, lines=FakeContent([])))
    counter = Counter()

    result = asyncio.run(
        load_tested.send_stream_request(
            session, "http://localhost/v1/chat/completions", "model", "prompt", counter
        )
    )

    assert result is None
    assert counter == {"http_503": 1}


def test_benchmark_failed_run_keeps_full_metrics_schema(tmp_path, monkeypatch):
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text('{"text": "one"}\n', encoding="utf-8")

    class EmptySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr(load_tested.aiohttp, "ClientSession", EmptySession)
    monkeypatch.setattr(load_tested, "init_nvml", lambda: False)

    async def failed_request(*_args, **_kwargs):
        return None

    monkeypatch.setattr(load_tested, "send_stream_request", failed_request)
    metrics = asyncio.run(
        load_tested.benchmark(
            endpoint="http://localhost/v1/chat/completions",
            model="model",
            dataset_path=str(dataset),
            concurrency=1,
            num_requests=1,
            warmup=0,
            tag="failed-run",
            max_tokens=4,
            ignore_eos=False,
            gpu_index=0,
        )
    )

    assert metrics["total_requests"] == 1
    assert metrics["successful_requests"] == 0
    assert metrics["throughput_req_per_sec"] is None
    assert metrics["ttft_p50_ms"] is None
    assert metrics["errors"] == {}
    assert "vram_samples_collected" in metrics


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
