# Python binding

This is a small reference binding implemented with `ctypes`.

```bash
cargo build -p kestrel-ffi
PYTHONPATH=bindings/python python3 bindings/python/example.py
PYTHONPATH=bindings/python python3 bindings/python/example_async.py
PYTHONPATH=bindings/python python3 bindings/python/example_persistence.py
PYTHONPATH=bindings/python python3 bindings/python/example_stream_model.py
PYTHONPATH=bindings/python python3 bindings/python/example_resume.py
```

Set `KESTREL_FFI_PATH` when the shared library is not under `target/debug` or
`target/release`.

The example uses a Python model callback and a Python echo tool, so it runs
offline. For the OpenAI path:

```python
from kestrel import Agent

with Agent({
    "system_prompt": "Be concise.",
    "streaming": True,
    "openai": {"model": "gpt-4.1-mini"},
}) as agent:
    agent.set_event_handler(lambda event: print(event))
    output = agent.run("Hello")
    print(output["text"])
```

The native run is blocking. In an async Python application, call it through
`asyncio.to_thread` or your framework's equivalent worker mechanism.

For native asyncio integration, prefer the nonblocking run handle:

```python
run = agent.start_run("Hello")
event = await run.next_event_async(1000)
output = await run.wait_async()
run.close()
```

`Agent.close()` and `Run.close()` wait for active runs and in-flight native
callbacks. Do not call them from inside an event/tool/model callback; in async
code, move blocking close/wait operations to `asyncio.to_thread`.

`Agent(..., session_store=..., memory=...)` installs host-backed persistence.
Objects follow the same protocols as the Rust traits:

- session store: `load(id) -> dict | None`, `save(session)`, optional `delete(id)`
- memory: `recall(query) -> list[dict]`, `remember(entry)`

Use `agent.reset()` to clear history while preserving the session ID, or
`agent.delete_session()` to remove the stored session as well.

`agent.resume()` continues an interrupted run. `agent.start_resume()` returns
the same pollable `Run` handle as `start_run()`. Pass
`retry_uncertain_tools=True` only when re-running an interrupted tool is safe.

See `example_persistence.py` for an in-memory database example.

Contract helpers mirror the C ABI:

```python
import kestrel

kestrel.contract_version()          # 1
kestrel.capabilities()              # capability names
kestrel.validate_config(config, has_custom_model=True)
```

Custom Python models can stream tokens by accepting an `emit` function:

```python
def stream_model(request, emit):
    emit({"type": "text", "text": "hello "})
    emit({"type": "text", "text": "world"})
    return {...}  # complete ModelResponse

Agent(config, stream_model=stream_model)
```
