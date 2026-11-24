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

module.exports = require(destination);
