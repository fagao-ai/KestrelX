"""Host-backed SessionStore and Memory example."""

from __future__ import annotations

from typing import Any, Optional

from kestrel import Agent


class DictSessionStore:
    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}

    def load(self, session_id: str) -> Optional[dict[str, Any]]:
        return self.sessions.get(session_id)

    def save(self, session: dict[str, Any]) -> None:
        self.sessions[session["id"]] = session

    def delete(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)


class ListMemory:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = [
            {
                "id": "memory-1",
                "text": "The user's name is Ada",
                "tags": ["profile"],
                "metadata": {},
                "created_at_ms": 1,
            }
        ]

    def recall(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        return self.entries

    def remember(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)


def make_model(requests: list[dict[str, Any]]):
    def model(request: dict[str, Any]) -> dict[str, Any]:
        requests.append(request)
        return {
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": f"answer #{len(requests)}"}],
            },
            "finish_reason": "stop",
            "usage": {"input_tokens": 3, "output_tokens": 2},
            "model": "python-persistence-callback",
        }

    return model


def main() -> None:
    sessions = DictSessionStore()
    memory = ListMemory()
    requests: list[dict[str, Any]] = []
    config = {
        "session_id": "python-persisted-session",
        "streaming": False,
        "memory": {"enabled": True},
        "openai": {"model": "python-persistence-callback"},
    }

    with Agent(
        config,
        model=make_model(requests),
        session_store=sessions,
        memory=memory,
    ) as agent:
        print(agent.run("first prompt")["text"])

    with Agent(
        config,
        model=make_model(requests),
        session_store=sessions,
        memory=memory,
    ) as agent:
        print(agent.run("second prompt")["text"])

    history = str(requests[-1])
    assert "first prompt" in history
    assert "Ada" in history
    print("persisted history and memory recall verified")


if __name__ == "__main__":
    main()
