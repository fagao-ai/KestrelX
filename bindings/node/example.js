'use strict';

const { createAgent } = require('./index.js');

async function main() {
  let calls = 0;
  const model = async (request) => {
    calls += 1;
    if (calls === 1) {
      return {
        message: {
          role: 'assistant',
          tool_calls: [
            {
              id: 'call-1',
              name: 'echo',
              arguments: { text: 'hello from Node' },
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
        content: [{ type: 'text', text: `result: ${tool.content[0].text}` }],
      },
      finish_reason: 'stop',
    };
  };

  const agent = createAgent(
    { streaming: true, openai: { model: 'node-model' } },
    model,
    (event) => {
      if (event.type === 'model_delta') {
        process.stdout.write(event.delta.text);
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
    })
  );

  await agent.run('use echo');
  console.log();
  agent.close();
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
