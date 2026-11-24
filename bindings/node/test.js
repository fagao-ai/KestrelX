'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const {
  capabilities,
  contractVersion,
  createAgent,
  validateConfig,
} = require('./index.js');

test('contract version and capabilities', () => {
  assert.equal(contractVersion(), 1);
  assert.ok(capabilities().includes('streaming'));
  assert.ok(capabilities().includes('session_store'));
  assert.ok(capabilities().includes('memory_store'));
  assert.ok(!capabilities().includes('mcp_http'));
});

test('rejects unsupported MCP config', () => {
  assert.throws(
    () =>
      createAgent({
        schema_version: 1,
        openai: { model: 'unused' },
        mcp_servers: [{ url: 'https://example.com/mcp' }],
      }),
    /mcp_http/
  );
});

test('validates shared fixture', () => {
  const fixture = JSON.parse(
    fs.readFileSync(
      path.join(__dirname, '../../kestrel-contract/fixtures/minimal.json'),
      'utf8'
    )
  );
  validateConfig(fixture, false);
});

test('JavaScript model, tool, and streaming events', async () => {
  let calls = 0;
  const deltas = [];
  const model = async (request) => {
    calls += 1;
    if (calls === 1) {
      return {
        message: {
          role: 'assistant',
          tool_calls: [
            {
              id: 'node-call-1',
              name: 'echo',
              arguments: { text: 'hello' },
            },
          ],
        },
        finish_reason: 'tool_calls',
      };
    }
    const toolMessage = request.messages.findLast(
      (message) => message.role === 'tool'
    );
    return {
      message: {
        role: 'assistant',
        content: [
          {
            type: 'text',
            text: `final: ${toolMessage.content[0].text}`,
          },
        ],
      },
      finish_reason: 'stop',
      usage: { input_tokens: 3, output_tokens: 4 },
    };
  };

  const agent = createAgent(
    { streaming: true, openai: { model: 'node-js-model' } },
    model,
    (event) => {
      if (event.type === 'model_delta' && event.delta?.type === 'text') {
        deltas.push(event.delta.text);
      }
    }
  );

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
    async (_callId, arguments_) => ({
      content: `echo:${arguments_.text}`,
      is_error: false,
    })
  );

  const output = await agent.run('use the tool');
  assert.equal(output.text, 'final: echo:hello');
  assert.deepEqual(deltas, ['final: echo:hello']);

  const session = await agent.session();
  assert.equal(session.messages.length, 4);
  assert.equal(session.messages[2].role, 'tool');
  agent.close();
});

test('custom JavaScript streamModel emits real-time deltas', async () => {
  const deltas = [];
  const agent = createAgent(
    { streaming: true, openai: { model: 'js-stream-model' } },
    {
      streamModel: async (_request, emit) => {
        emit({ type: 'text', text: 'one ' });
        emit({ type: 'text', text: 'two' });
        return {
          message: {
            role: 'assistant',
            content: [{ type: 'text', text: 'one two' }],
          },
          finish_reason: 'stop',
        };
      },
      onEvent: (event) => {
        if (event.type === 'model_delta') {
          deltas.push(event.delta.text);
        }
      },
    }
  );
  const output = await agent.run('stream');
  assert.equal(output.text, 'one two');
  assert.deepEqual(deltas, ['one ', 'two']);
  agent.close();
});

test('JS SessionStore and Memory callbacks', async () => {
  const sessions = new Map();
  const sessionStore = {
    load: (id) => sessions.get(id) ?? null,
    save: (session) => {
      sessions.set(session.id, session);
    },
    delete: (id) => {
      sessions.delete(id);
    },
  };
  const memoryEntries = [
    { id: 'memory-1', text: "The user's name is Ada", tags: ['profile'] },
  ];
  const memory = {
    recall: () => memoryEntries,
    remember: (entry) => {
      memoryEntries.push(entry);
    },
  };
  const requests = [];
  let calls = 0;
  const model = async (request) => {
    calls += 1;
    requests.push(request);
    const text = calls === 1 ? 'first answer' : 'second answer';
    return {
      message: {
        role: 'assistant',
        content: [{ type: 'text', text }],
      },
      finish_reason: 'stop',
    };
  };
  const config = {
    session_id: 'node-persisted-session',
    streaming: true,
    openai: { model: 'js-persisted-model' },
    memory: { tools: true },
  };

  const first = createAgent(config, { model, sessionStore, memory });
  assert.equal((await first.run('first prompt')).text, 'first answer');
  first.close();
  assert.ok(sessions.get('node-persisted-session'));

  const second = createAgent(config, { model, sessionStore, memory });
  assert.equal((await second.run('second prompt')).text, 'second answer');
  second.close();

  const history = JSON.stringify(requests.at(-1));
  assert.match(history, /first prompt/);
  assert.match(history, /first answer/);
  assert.match(history, /second prompt/);
  assert.match(history, /Ada/);
});

test('startRun returns a cancellable handle', async () => {
  let notifyStarted;
  const started = new Promise((resolve) => {
    notifyStarted = resolve;
  });
  const model = async () => {
    notifyStarted();
    return new Promise(() => {});
  };
  const agent = createAgent(
    { streaming: true, openai: { model: 'slow-js-model' } },
    { model }
  );

  const run = agent.startRun('wait forever');
  assert.equal(run.isDone(), false);
  await started;
  run.cancel();
  await assert.rejects(run.result(), /cancelled/);
  assert.equal(run.isDone(), true);
  agent.close();
});

test('startRun resolves its output once', async () => {
  const agent = createAgent(
    { streaming: false, openai: { model: 'fast-js-model' } },
    {
      model: async () => ({
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'finished' }],
        },
        finish_reason: 'stop',
      }),
    }
  );
  const run = agent.startRun('finish');
  const output = await run.result();
  assert.equal(output.text, 'finished');
  await assert.rejects(run.result(), /already been consumed/);
  agent.close();
});
