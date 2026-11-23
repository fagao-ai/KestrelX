# Python binding

This is a small reference binding implemented with `ctypes`.

```bash
cargo build -p kestrel-ffi
PYTHONPATH=bindings/python python3 bindings/python/example.py
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
