# Go binding

This package uses cgo to call the `kestrel-ffi` C ABI directly. It has no
third-party Go dependencies.

```bash
cargo build -p kestrel-ffi
cd bindings/go
go test ./...
go run ./examples/basic
```

The default cgo link flags point to `target/debug`. For release applications,
build `cargo build -p kestrel-ffi --release` and adjust the `#cgo LDFLAGS`
paths in `callbacks.go`, or provide an equivalent deployment layout.

## Features

- OpenAI-compatible config passthrough
- custom model and streaming model callbacks
- synchronous tools
- event callbacks and per-run event polling
- blocking `Run(ctx)` and nonblocking `StartRun` / `Run.Wait`
- cancellation through `context.Context`
- SessionStore and long-term Memory callbacks
- MCP, Skills, Todo, sub-agent, memory, and compression through config JSON

## Example

```go
agent, err := kestrel.New(
    map[string]any{
        "streaming": true,
        "openai": map[string]any{"model": "gpt-4.1-mini"},
    },
)
if err != nil {
    log.Fatal(err)
}
defer agent.Close()

output, err := agent.Run(context.Background(), "Hello")
```

For custom models:

```go
agent, err := kestrel.New(
    map[string]any{"streaming": true},
    kestrel.WithStreamingModel(func(
        request kestrel.ModelRequest,
        emit func(kestrel.ModelDelta),
    ) (kestrel.ModelResponse, error) {
        emit(kestrel.ModelDelta{Type: "text", Text: "hello"})
        return kestrel.ModelResponse{
            Message: kestrel.Message{
                Role: kestrel.RoleAssistant,
                Content: []kestrel.ContentPart{kestrel.TextPart("hello")},
            },
            FinishReason: "stop",
        }, nil
    }),
)
```

## Callback constraints

- Event, model, and tool callbacks run synchronously on native threads.
- Do not free or close an agent from inside its callbacks.
- A tool callback cannot be interrupted after it starts; `context.Context`
  cancellation is observed between model/tool steps and while polling runs.
- Callback panics are recovered and converted into model/tool errors.
