"""Custom Python model with token-level FFI deltas."""

from __future__ import annotations

from typing import Any, Callable

from kestrel import Agent


def stream_model(
    request: dict[str, Any],
    emit: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    del request
    for text in ["Python ", "streams ", "tokens"]:
        emit({"type": "text", "text": text})
    return {
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "Python streams tokens"}],
        },
        "finish_reason": "stop",
        "usage": {"input_tokens": 2, "output_tokens": 5},
        "model": "python-streaming-model",
    }


def main() -> None:
    deltas: list[str] = []
    config = {
        "streaming": True,
        "openai": {"model": "python-streaming-model"},
    }
    with Agent(config, stream_model=stream_model) as agent:
        agent.set_event_handler(
            lambda event: deltas.append(event["delta"]["text"])
            if event["type"] == "model_delta"
            else None
        )
        output = agent.run("stream")
        print(output["text"])
        assert deltas == ["Python ", "streams ", "tokens"]


if __name__ == "__main__":
    main()
