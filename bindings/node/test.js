'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const { createAgent } = require('./index.js');

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
