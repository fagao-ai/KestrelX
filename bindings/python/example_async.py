"""Async polling example for the Python binding."""

from __future__ import annotations

import asyncio

from kestrelx import Agent


def model(request: dict) -> dict:
    last_user = next(
        message
        for message in reversed(request["messages"])
        if message["role"] == "user"
    )
    return {
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"async model saw: {last_user['content'][0]['text']}",
                }
            ],
        },
        "finish_reason": "stop",
        "usage": {"input_tokens": 4, "output_tokens": 6},
        "model": "python-async-callback",
    }


async def main() -> None:
    config = {
        "streaming": True,
        "openai": {"model": "python-async-callback"},
    }
    with Agent(config, model=model) as agent:
        run = agent.start_run("hello from asyncio")
        try:
            while True:
                event = await run.next_event_async(1_000)
                if event is None:
                    if run.is_done:
                        break
                    continue
                if event["type"] == "model_delta":
                    print(event["delta"]["text"])
                if event["type"] in {"run_completed", "run_failed"}:
                    break
            output = await run.wait_async()
            print(output["text"])
        finally:
            run.close()


if __name__ == "__main__":
    asyncio.run(main())
