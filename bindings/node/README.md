# Node.js native addon

This package uses napi-rs to expose Kestrel directly as a native Node addon.
There is no Koffi or blocking FFI bridge in the run path.

```bash
cargo build -p kestrel-node
cd bindings/node
node --test
node example.js
```

`index.js` copies the platform artifact from `target/debug` to `native.node`
and loads it. Set `KESTREL_NODE_PROFILE=release` to load `target/release`
instead.

## API

```js
const { createAgent } = require('./index.js');

const agent = createAgent(
  {
    schema_version: 1,
    system_prompt: 'Be concise.',
    streaming: true,
    openai: { model: 'gpt-4.1-mini' },
  },
  undefined,
  (event) => {
    if (event.type === 'model_delta') process.stdout.write(event.delta.text);
  }
);

const output = await agent.run('Hello');
agent.close();
```

For cancellation, start a run without blocking:

```js
const run = agent.startRun('Do a long task');
// Later, from any callback or timer:
run.cancel();

try {
  const output = await run.result();
} catch (error) {
  // Cancelled runs reject with "agent run cancelled".
}
```

`startRun()` returns immediately. `run.isDone()` checks native completion and
`run.result()` can be consumed once.

Custom JavaScript models are async functions returning a serialized
`ModelResponse` shape:

```js
const model = async (request) => ({
  message: {
    role: 'assistant',
    content: [{ type: 'text', text: 'hello' }],
  },
  finish_reason: 'stop',
});
```

Custom models can emit real-time deltas:

```js
const agent = createAgent(config, {
  streamModel: async (request, emit) => {
    emit({ type: 'text', text: 'hello ' });
    emit({ type: 'text', text: 'world' });
    return response;
  },
  onEvent: (event) => {
    if (event.type === 'model_delta') console.log(event.delta);
  },
});
```

Non-streaming `model` handlers emit their final assistant text as one delta.
The built-in OpenAI provider emits native token/tool-call deltas.

Tools are async functions receiving `(callId, arguments)` and returning either
a string or a `ToolOutput` object:

```js
agent.registerTool(
  {
    name: 'echo',
    description: 'Echo text',
    input_schema: {
      type: 'object',
      properties: { text: { type: 'string' } },
      required: ['text'],
    },
  },
  async (_callId, args) => ({ content: `echo:${args.text}` })
);
```

TypeScript declarations are in `index.d.ts`.

## Host persistence

The options object can provide JS-backed session and long-term memory stores:

```js
const agent = createAgent(config, {
  model,
  sessionStore: {
    load: async (id) => database.load(id), // object or null
    save: async (session) => database.save(session),
    delete: async (id) => database.delete(id),
  },
  memory: {
    recall: async (query) => database.recall(query), // array of entries
    remember: async (entry) => database.remember(entry),
  },
});
```

Payloads cross the native boundary as JSON strings internally so 64-bit
timestamps remain lossless.

The addon exports `contractVersion()`, `capabilities()`, and
`validateConfig(config, hasCustomModel?)`. It shares `kestrel-contract` with the
Rust and FFI SDKs; unsupported config is rejected before an Agent is created.

## Current scope

- OpenAI-compatible provider through config
- custom async JavaScript model
- custom JavaScript streaming model
- async JavaScript tools
- event callback with OpenAI token stream
- Todo, in-memory memory tools, compression, Skills, and sub-agents via config
- session/todo/reset snapshots
- JS-backed SessionStore and Memory
- cancellable per-run handles via `startRun()`

Not yet exposed by the Node addon: MCP configuration. Unsupported config fields
and capabilities are rejected instead of silently ignored.

## Threading

Napi ThreadsafeFunctions keep JavaScript callbacks on the JS thread while the
Rust Tokio runtime executes the agent. Always call `agent.close()` after all
`run()` promises and run handles settle so callback references and native
resources are released.
