# kestrel-contract

Shared, versioned JSON configuration for Kestrel language bindings.

```json
{
  "schema_version": 1,
  "openai": { "model": "gpt-4.1-mini" }
}
```

`schema_version` defaults to `1`. Unknown fields are rejected. A binding parses
an [`AgentConfig`](src/config.rs), calls `validate(has_custom_model,
capabilities)`, and only then constructs an agent.

## Capabilities

The contract returns required capabilities from the config and compares them
with what a binding implements. Examples:

- `openai`, `custom_model`, `streaming`
- `todo`, `memory`, `subagent`, `skills`
- `compression_sliding`, `compression_summarizing`
- `mcp_stdio`, `mcp_http`
- `host_tools`, `session_store`, `memory_store`

This prevents a binding from silently ignoring config it cannot implement. For
example, the current Node addon accepts the shared schema but rejects
`mcp_servers` because it does not yet advertise `mcp_stdio` / `mcp_http`.

## Golden fixtures

`fixtures/minimal.json` and `fixtures/full.json` are consumed by Rust, FFI,
Python, Go, and Node tests. Add fixtures for every wire-visible behavior and
update capability lists deliberately.
