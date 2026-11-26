"""Resume a checkpointed run through the Python binding."""

from __future__ import annotations

from typing import Any, Optional

from kestrel_agent_core import Agent


class ResumeStore:
    def __init__(self, session: dict[str, Any]) -> None:
        self.sessions = {session["id"]: session}

    def load(self, session_id: str) -> Optional[dict[str, Any]]:
        return self.sessions.get(session_id)

    def save(self, session: dict[str, Any]) -> None:
        self.sessions[session["id"]] = session

    def delete(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)


def checkpoint(run_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": "python-resume",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "finish the interrupted run"}],
            }
        ],
        "metadata": {},
        "created_at_ms": 1,
        "updated_at_ms": 1,
        "revision": 1,
        "active_run": {
            "run_id": run_id,
            "turn": 1,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "phase": {"type": "model_call"},
            "updated_at_ms": 1,
        },
    }


def make_model(counter: list[int]):
    def model(_request: dict[str, Any]) -> dict[str, Any]:
        counter[0] += 1
        return {
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"resumed #{counter[0]}"}
                ],
            },
            "finish_reason": "stop",
        }

    return model


def main() -> None:
    config = {
        "session_id": "python-resume",
        "streaming": False,
        "openai": {"model": "python-resume-model"},
    }
    counter = [0]

    store = ResumeStore(checkpoint("python-run-1"))
    with Agent(config, model=make_model(counter), session_store=store) as agent:
        output = agent.resume()
        assert output["run_id"] == "python-run-1"
        assert output["text"] == "resumed #1"
        assert "active_run" not in store.sessions["python-resume"]

    store = ResumeStore(checkpoint("python-run-2"))
    with Agent(config, model=make_model(counter), session_store=store) as agent:
        run = agent.start_resume()
        try:
            output = run.wait()
        finally:
            run.close()
        assert output["run_id"] == "python-run-2"
        assert output["text"] == "resumed #2"

    print("resume verified")


if __name__ == "__main__":
    main()
