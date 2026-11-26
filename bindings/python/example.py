"""Offline example using a Python model callback."""

from __future__ import annotations

from kestrelx import Agent


def model(request: dict) -> dict:
    tool_results = [
        message
        for message in request["messages"]
        if message["role"] == "tool"
    ]
    if not tool_results:
        return {
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "python-call-1",
                        "name": "echo",
                        "arguments": {"text": "hello from Python"},
                    }
                ],
            },
            "finish_reason": "tool_calls",
            "usage": {"input_tokens": 5, "output_tokens": 3},
            "model": "python-callback",
        }

    tool_text = tool_results[-1]["content"][0]["text"]
    return {
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": f"Python model received: {tool_text}"}],
        },
        "finish_reason": "stop",
        "usage": {"input_tokens": 5, "output_tokens": 7},
        "model": "python-callback",
    }


def main() -> None:
    config = {
        "max_turns": 3,
        "streaming": True,
        "openai": {"model": "python-callback"},
    }
    with Agent(config, model=model) as agent:
        agent.register_tool(
            "echo",
            "Echo supplied text.",
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            lambda arguments, _call_id: {
                "content": f"echo:{arguments['text']}",
                "is_error": False,
            },
        )
        agent.set_event_handler(
            lambda event: print(event) if event["type"] == "model_delta" else None
        )
        output = agent.run("hello from Python")
        print(output["text"])


if __name__ == "__main__":
    main()
