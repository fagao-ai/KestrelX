"""Smoke test an installed kestrel-agent-core wheel without network access."""

from __future__ import annotations

import kestrel_agent_core as kestrel


def model(_request: dict) -> dict:
    return {
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "installed wheel works"}],
        },
        "finish_reason": "stop",
    }


def main() -> None:
    assert kestrel.__version__
    assert kestrel.contract_version() == 1
    capabilities = set(kestrel.capabilities())
    assert {"openai_responses", "anthropic", "gemini"} <= capabilities

    for kind in ("openai_chat", "openai_responses", "anthropic", "gemini"):
        with kestrel.Agent(
            {
                "schema_version": 1,
                "provider": {"kind": kind, "model": "test-model"},
            }
        ):
            pass

    with kestrel.Agent(
        {"schema_version": 1, "streaming": False},
        model=model,
    ) as agent:
        output = agent.run("hello")
    assert output["text"] == "installed wheel works"
    print(f"kestrel-agent-core {kestrel.__version__}: wheel smoke test passed")


if __name__ == "__main__":
    main()
