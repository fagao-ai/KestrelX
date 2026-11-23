# kestrel-ffi

`kestrel-ffi` exposes the Kestrel agent runtime through a JSON-first C ABI. It is
meant to be the base layer for Python, Node, Go, Ruby, JVM, or any other SDK.

```bash
cargo build -p kestrel-ffi --release
```

Artifacts:

- `libkestrel_ffi.dylib` / `.so` / `kestrel_ffi.dll`
- static library and rlib
- header: [`include/kestrel.h`](include/kestrel.h)

## API surface

The opaque `KestrelAgent` handle owns:

- the Rust agent and its conversation
- an internal multithreaded Tokio runtime
- an event callback slot
- cancellation state

Main functions:

| Function | Purpose |
|---|---|
| `kestrel_agent_new` | OpenAI-compatible agent from config JSON |
| `kestrel_agent_new_with_model` | agent backed by a host model callback |
| `kestrel_agent_register_tool` | register a synchronous host tool |
| `kestrel_agent_set_event_callback` | receive `AgentEvent` JSON |
| `kestrel_agent_run` | bidirectional blocking run for text input |
| `kestrel_agent_run_message` | same, with a full `Message` JSON input |
| `kestrel_agent_builder_*` | builder with host SessionStore/Memory callbacks |
| `kestrel_agent_start_run` | nonblocking run handle |
| `kestrel_run_next_event` | poll the per-run event queue |
| `kestrel_run_wait` | wait for final output without blocking the start |
| `kestrel_run_is_done` / `kestrel_run_cancel` / `kestrel_run_free` | run lifecycle |
| `kestrel_agent_cancel` | cancel the active run from any thread |
| `kestrel_agent_session_json` | conversation snapshot |
| `kestrel_agent_todos_json` | todo snapshot when enabled |
| `kestrel_agent_reset` | clear history while retaining session ID |

All JSON-returning functions return a newly allocated C string on success. Free
it with `kestrel_string_free`. Errors use a `char **error` out-parameter and must
also be freed with `kestrel_string_free`.

## Configuration

```json
{
  "system_prompt": "You are a concise assistant.",
  "prompt_sections": ["Prefer short answers."],
  "session_id": "user-42",
  "max_turns": 16,
  "temperature": 0.2,
  "max_output_tokens": 2048,
  "context_window": 128000,
  "reserve_output_tokens": 4096,
  "streaming": true,
  "fail_on_tool_error": false,
  "tool_choice": "auto",
  "todo": true,
  "memory": {
    "enabled": true,
    "tools": true,
    "auto_recall": true,
    "recall_limit": 5,
    "max_context_tokens": 2000
  },
  "compression": {
    "kind": "sliding",
    "max_tokens": 80000
  },
  "subagent": {
    "name": "delegate",
    "description": "Delegate a self-contained task.",
    "max_depth": 2
  },
  "skills": {
    "paths": ["./skills"],
    "max_depth": 4
  },
  "mcp_servers": [
    {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "env": {},
      "cwd": ".",
      "request_timeout_seconds": 30,
      "protocol_version": "2025-06-18",
      "client_name": "kestrel"
    }
  ],
  "openai": {
    "model": "gpt-4.1-mini",
    "api_key": "optional",
    "base_url": "optional",
    "context_window": 128000,
    "timeout_seconds": 120,
    "stream_usage": true
  }
}
```

`compression.kind` accepts `sliding` or `summarizing`. OpenAI credentials fall
back to `OPENAI_API_KEY` and `OPENAI_BASE_URL`.

`skills.paths` may contain multiple directories; duplicate skill names are
rejected. `mcp_servers` only supports stdio in the current release and connects
during agent construction. Agent construction with MCP blocks and must not run
on a Tokio runtime thread.

For `kestrel_agent_new_with_model`, `openai.model` is only used as the model's
display name; no HTTP request is made.

## Host persistence

Use `kestrel_agent_builder_new` or `kestrel_agent_builder_new_with_model`, then
register persistence before building:

```c
KestrelAgentBuilder *builder = kestrel_agent_builder_new(config_json, &error);
kestrel_agent_builder_set_session_store(
    builder, load_session, save_session, delete_session, store_ctx, &error);
kestrel_agent_builder_set_memory(
    builder, recall_memory, remember_memory, memory_ctx, &error);
KestrelAgent *agent = kestrel_agent_builder_build(builder, &error);
kestrel_agent_builder_free(builder);
```

Session callbacks:

- load: return `Session` JSON, or NULL when not found
- save: return 0 on success
- delete: optional

Memory callbacks:

- recall: receive `MemoryQuery` JSON and return an array of `MemoryEntry` JSON
- remember: receive `MemoryEntry` JSON and return 0 on success

All callback-returned JSON and error strings must be allocated with
`kestrel_string_new`; Kestrel takes ownership. Do not nest a builder build.

## Callback contracts

### Tool callback

```c
char *tool_callback(void *user_data, const char *call_id,
                    const char *arguments_json);
```

Return either:

- plain text, or
- `ToolOutput` JSON such as `{"content":"ok","is_error":false}`.

The return value must be allocated with `kestrel_string_new`; Kestrel takes
ownership and frees it. The callback runs on a Tokio blocking-pool thread and
must be thread-safe.

### Model callback

```c
char *model_callback(void *user_data, const char *request_json);
```

`request_json` is a serialized `kestrel::ModelRequest`. Return a serialized
`kestrel::ModelResponse`, allocated with `kestrel_string_new`.

The custom callback path currently emits the final assistant text as one
`model_delta` event. Native token-level streaming is implemented by the
OpenAI-compatible provider.

### Event callback

```c
void event_callback(void *user_data, const char *event_json);
```

The string is borrowed and valid only for the duration of the callback. Copy it
before returning. Callbacks must be fast, must not block, and must not unwind.

Events include:

- `run_started`, `turn_started`
- `model_delta` with `ModelDelta::Text` or `ModelDelta::ToolCall`
- `model_completed`
- `tool_started`, `tool_completed`
- `memory_recalled`, `context_compressed`
- `run_completed`, `run_failed`

## Async run handles

`kestrel_agent_start_run*` returns immediately with a `KestrelRun`. The agent
accepts only one active run at a time, but the host can keep processing its own
event loop while the Rust runtime executes.

`kestrel_run_next_event` returns:

- `1`: an event JSON string was written
- `0`: timeout
- `2`: the queue has ended
- `-1`: FFI error

`kestrel_run_wait` returns:

- `1`: final `AgentOutput` JSON was written
- `0`: timeout
- `-1`: run or FFI error

`timeout_ms < 0` waits indefinitely. The event queue is unbounded; SDKs should
continuously drain it or rely on the global event callback instead.

Run-level event delivery still forwards to the agent event callback, while also
populating the per-run poll queue. `kestrel_run_free` cancels and waits for the
run and any in-flight host callbacks before returning.

## Threading and lifetime rules

- `kestrel_agent_run*` is synchronous and internally calls a blocking Tokio
  runtime. Do not call it from a Tokio runtime thread.
- `kestrel_agent_start_run*` is nonblocking, but `kestrel_run_next_event` and
  `kestrel_run_wait` are blocking C calls.
- An agent should outlive all of its async run handles.
- Agent construction with `mcp_servers` blocks while initializing MCP processes.
- Native SDKs should execute the blocking call on a worker thread.
- `kestrel_agent_cancel` is thread-safe.
- One agent accepts one active run at a time.
- `user_data` pointers must stay valid until `kestrel_agent_free`.
- Do not free an agent while a run is active.
- Do not block indefinitely in an event callback.
- Do not replace the event callback or free its `user_data` from inside the
  callback itself.
- A synchronous tool/model callback cannot be interrupted once entered. Cancel
  takes effect at the next await/check point, but the blocking run waits for
  already-started host callbacks to return before reporting cancellation. This
  keeps `user_data` valid for the full callback lifetime.

## Python reference binding

```bash
cargo build -p kestrel-ffi
PYTHONPATH=bindings/python python3 bindings/python/example.py
PYTHONPATH=bindings/python python3 bindings/python/example_async.py
PYTHONPATH=bindings/python python3 bindings/python/example_persistence.py
```

The ctypes wrapper supports OpenAI config, custom model callbacks, tools, events,
session snapshots, todos, reset, cancellation, and `start_run` /
`next_event_async` / `wait_async`, plus host-backed session and memory objects.

## Current limits

- MCP is stdio-only and does not expose HTTP transport, sampling, or roots.
- Skills use the core flat frontmatter parser.
- If no host SessionStore/Memory callbacks are registered, state remains
  in-memory for the lifetime of the agent handle.
- Custom model callbacks are non-token-streaming.
- Event delivery is synchronous callback delivery, not a bounded queue.
