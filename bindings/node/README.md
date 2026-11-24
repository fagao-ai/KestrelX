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

Custom models currently emit their final text as one delta. The built-in
OpenAI provider emits real token/tool-call deltas through the event handler.

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

The addon exports `contractVersion()`, `capabilities()`, and
`validateConfig(config, hasCustomModel?)`. It shares `kestrel-contract` with the
Rust and FFI SDKs; unsupported config is rejected before an Agent is created.

## Current scope

- OpenAI-compatible provider through config
- custom async JavaScript model
- async JavaScript tools
- event callback with OpenAI token stream
- Todo, in-memory memory tools, compression, Skills, and sub-agents via config
- session/todo/reset snapshots

Not yet exposed by the Node addon: JS-backed SessionStore/Memory callbacks,
MCP configuration, explicit per-run handles, and token streaming for custom
JavaScript models. Unsupported config fields are rejected instead of silently
ignored.

## Threading

Napi ThreadsafeFunctions keep JavaScript callbacks on the JS thread while the
Rust Tokio runtime executes the agent. Always call `agent.close()` after all
`run()` promises settle so callback references and native resources are
released.
