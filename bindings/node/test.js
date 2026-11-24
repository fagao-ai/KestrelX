'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const {
  capabilities,
  contractVersion,
  createAgent,
  createAgentAsync,
  validateConfig,
} = require('./index.js');

test('contract version and capabilities', () => {
  assert.equal(contractVersion(), 1);
  assert.ok(capabilities().includes('streaming'));
  assert.ok(capabilities().includes('session_store'));
  assert.ok(capabilities().includes('memory_store'));
  assert.ok(capabilities().includes('mcp_http'));
  assert.ok(capabilities().includes('mcp_stdio'));
});

test('sync createAgent directs MCP config to createAgentAsync', () => {
  assert.throws(
    () =>
      createAgent({
        schema_version: 1,
        openai: { model: 'unused' },
        mcp_servers: [{ url: 'https://example.com/mcp' }],
      }),
    /createAgentAsync/
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
  await second.deleteSession();
  assert.equal(sessions.has('node-persisted-session'), false);
  assert.equal((await second.session()).messages.length, 0);
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

test('createAgentAsync connects an HTTP MCP server', async () => {
  const http = require('node:http');
  const server = http.createServer(async (request, response) => {
    if (request.method === 'DELETE') {
      response.writeHead(204).end();
      return;
    }
    let body = '';
    for await (const chunk of request) body += chunk;
    const message = JSON.parse(body);
    if (message.id === undefined) {
      response.writeHead(202).end();
      return;
    }

    let result;
    if (message.method === 'initialize') {
      result = {
        protocolVersion: '2025-06-18',
        capabilities: { tools: {} },
        serverInfo: { name: 'node-test', version: '1.0.0' },
      };
    } else if (message.method === 'tools/list') {
      result = {
        tools: [
          {
            name: 'remote.echo',
            description: 'Echo through MCP',
            inputSchema: {
              type: 'object',
              properties: { text: { type: 'string' } },
              required: ['text'],
            },
          },
        ],
      };
    } else if (message.method === 'tools/call') {
      result = {
        content: [
          { type: 'text', text: `mcp:${message.params.arguments.text}` },
        ],
        isError: false,
      };
    } else {
      response
        .writeHead(200, { 'content-type': 'application/json' })
        .end(
          JSON.stringify({
            jsonrpc: '2.0',
            id: message.id,
            error: { code: -32601, message: 'unsupported' },
          })
        );
      return;
    }
    response
      .writeHead(200, {
        'content-type': 'application/json',
        'mcp-session-id': 'node-test-session',
      })
      .end(JSON.stringify({ jsonrpc: '2.0', id: message.id, result }));
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const url = `http://127.0.0.1:${server.address().port}/mcp`;

  let calls = 0;
  const model = async (request) => {
    calls += 1;
    if (calls === 1) {
      return {
        message: {
          role: 'assistant',
          tool_calls: [
            {
              id: 'mcp-call',
              name: 'remote_echo',
              arguments: { text: 'hello' },
            },
          ],
        },
        finish_reason: 'tool_calls',
      };
    }
    const tool = request.messages.findLast((message) => message.role === 'tool');
    return {
      message: {
        role: 'assistant',
        content: [{ type: 'text', text: tool.content[0].text }],
      },
      finish_reason: 'stop',
    };
  };

  const agent = await createAgentAsync(
    {
      streaming: false,
      mcp_servers: [{ url }],
      openai: { model: 'node-mcp-model' },
    },
    { model }
  );
  try {
    const output = await agent.run('use MCP');
    assert.equal(output.text, 'mcp:hello');

    await agent.closeMcp();
    calls = 0;
    const afterClose = await agent.run('after MCP close');
    assert.match(afterClose.text, /not registered/);
  } finally {
    await agent.closeMcp();
    agent.close();
    server.closeAllConnections?.();
    await new Promise((resolve) => server.close(resolve));
  }
});

test('createAgentAsync connects a stdio MCP server', async () => {
  const script = `
    const readline = require('node:readline');
    const rl = readline.createInterface({ input: process.stdin });
    rl.on('line', (line) => {
      const message = JSON.parse(line);
      if (message.id === undefined) return;
      let result;
      if (message.method === 'initialize') {
        result = {
          protocolVersion: '2025-06-18',
          capabilities: { tools: {} },
          serverInfo: { name: 'node-stdio-test', version: '1.0.0' },
        };
      } else if (message.method === 'tools/list') {
        result = {
          tools: [{
            name: 'stdio.echo',
            description: 'Echo through stdio MCP',
            inputSchema: {
              type: 'object',
              properties: { text: { type: 'string' } },
              required: ['text'],
            },
          }],
        };
      } else if (message.method === 'tools/call') {
        result = {
          content: [{ type: 'text', text: 'stdio:' + message.params.arguments.text }],
          isError: false,
        };
      } else {
        result = {};
      }
      process.stdout.write(JSON.stringify({
        jsonrpc: '2.0',
        id: message.id,
        result,
      }) + '\\n');
    });
  `;

  let calls = 0;
  const model = async (request) => {
    calls += 1;
    if (calls === 1) {
      return {
        message: {
          role: 'assistant',
          tool_calls: [
            {
              id: 'stdio-call',
              name: 'stdio_echo',
              arguments: { text: 'hello' },
            },
          ],
        },
        finish_reason: 'tool_calls',
      };
    }
    const tool = request.messages.findLast((message) => message.role === 'tool');
    return {
      message: {
        role: 'assistant',
        content: [{ type: 'text', text: tool.content[0].text }],
      },
      finish_reason: 'stop',
    };
  };

  const agent = await createAgentAsync(
    {
      streaming: false,
      mcp_servers: [
        {
          command: process.execPath,
          args: ['-e', script],
        },
      ],
      openai: { model: 'node-stdio-mcp-model' },
    },
    { model }
  );
  try {
    const output = await agent.run('use stdio MCP');
    assert.equal(output.text, 'stdio:hello');
  } finally {
    await agent.closeMcp();
    agent.close();
  }
});
