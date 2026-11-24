'use strict';

const fs = require('node:fs');
const path = require('node:path');

const profile = process.env.KESTREL_NODE_PROFILE || 'debug';
const repoRoot = path.resolve(__dirname, '../..');
const artifacts = {
  darwin: 'libkestrel_node.dylib',
  linux: 'libkestrel_node.so',
  win32: 'kestrel_node.dll',
};
const artifact = artifacts[process.platform];
if (!artifact) {
  throw new Error(`Unsupported platform: ${process.platform}`);
}

const source = path.join(repoRoot, 'target', profile, artifact);
const destination = path.join(__dirname, 'native.node');
if (!fs.existsSync(source)) {
  throw new Error(
    `Kestrel Node addon not found at ${source}. Run "cargo build -p kestrel-node".`
  );
}

const needsCopy =
  !fs.existsSync(destination) ||
  fs.statSync(source).mtimeMs > fs.statSync(destination).mtimeMs;
if (needsCopy) {
  fs.copyFileSync(source, destination);
}

const native = require(destination);

function extractText(message) {
  return (message?.content ?? [])
    .filter((part) => part.type === 'text')
    .map((part) => part.text)
    .join('\n');
}

function createAgent(config, modelOrOptions, eventHandler) {
  let options = {};
  if (typeof modelOrOptions === 'function' || modelOrOptions == null) {
    options.model = modelOrOptions;
    options.onEvent = eventHandler;
  } else {
    options = modelOrOptions;
  }
  if (options.model && options.streamModel) {
    throw new Error('model and streamModel are mutually exclusive');
  }

  let model;
  if (options.streamModel) {
    model = async (request, token) =>
      options.streamModel(request, (delta) =>
        native.emitModelDelta(token, delta)
      );
  } else if (options.model) {
    model = async (request, token) => {
      const response = await options.model(request);
      const text = extractText(response?.message);
      if (text) {
        native.emitModelDelta(token, { type: 'text', text });
      }
      return response;
    };
  }

  const store = options.sessionStore;
  const load = store
    ? async (id) => {
        const session = await store.load(id);
        return session == null ? null : JSON.stringify(session);
      }
    : undefined;
  const save = store
    ? async (sessionJson) => {
        await store.save(JSON.parse(sessionJson));
        return null;
      }
    : undefined;
  const remove = store?.delete
    ? async (id) => {
        await store.delete(id);
        return null;
      }
    : undefined;

  const memory = options.memory;
  const recall = memory
    ? async (queryJson) =>
        JSON.stringify((await memory.recall(JSON.parse(queryJson))) ?? [])
    : undefined;
  const remember = memory
    ? async (entryJson) => {
        await memory.remember(JSON.parse(entryJson));
        return null;
      }
    : undefined;

  return native.createAgent(
    config,
    model,
    options.onEvent ?? options.eventHandler,
    load,
    save,
    remove,
    recall,
    remember
  );
}

module.exports = {
  ...native,
  createAgent,
};
