# Kestrel

Kestrel 是一个场景无关、库优先的 Rust agent core。它不假设你在做编程助手、办公助手或客服机器人，只提供可组合的运行时原语：

- LLM 抽象与 OpenAI-compatible 实现
- 带取消、token 流、事件流和错误恢复的 agent loop
- 工具注册与 JSON Schema 工具调用
- 上下文压缩（滑动窗口 / LLM 摘要）
- 会话持久化与长期记忆
- 可选 Todo、子 agent、MCP、Skills
- C ABI 与 Python/Go/Node 参考绑定

目标不是再做一个“什么都能做”的完整应用，而是让开发者用同一套内核构建自己的产品。

## 设计原则

1. **核心不认识业务场景。** 文件、Shell、浏览器、CRM 等能力都应作为 Tool 注入。
2. **依赖倒置。** Agent 只依赖 `Model`、`Tool`、`ContextCompressor`、`Memory`、`SessionStore` 和 `EventSink`。
3. **扩展不侵入主循环。** Todo、子 agent、MCP、Skills 通过工具、提示片段或独立模块接入。
4. **失败可恢复。** 工具错误默认回传给模型；模型错误、取消、turn limit、上下文溢出有明确类型。
5. **SDK 友好。** 消息、工具描述、输出和事件全部可序列化，提供稳定 JSON + C ABI 边界。
6. **文件保持可读。** 所有 Rust 源文件均低于 1200 行，模块按职责拆分。

## 快速开始

```toml
[dependencies]
kestrel = { path = "." }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use std::sync::Arc;
use kestrel::{Agent, OpenAiCompatible, SlidingWindowCompressor};

#[tokio::main]
async fn main() -> kestrel::Result<()> {
    let model = OpenAiCompatible::from_env("gpt-4.1-mini")?;

    let agent = Agent::builder(model)
        .system_prompt("You are a concise assistant.")
        .compressor(Arc::new(SlidingWindowCompressor::new(100_000)))
        .max_turns(8)
        .build()?;

    let output = agent.run("用一句话解释 agent loop").await?;
    println!("{}", output.text);
    Ok(())
}
```

设置 `OPENAI_API_KEY`；本地兼容服务可通过 `OPENAI_BASE_URL` 指向。完整示例见
[`examples/support_agent.rs`](examples/support_agent.rs)。

## 核心 API

### Model

`Model` 是唯一的 LLM 边界。厂商 SDK、HTTP API、本地推理都实现这个 trait：

```rust
use async_trait::async_trait;
use kestrel::{Model, ModelRequest, ModelResponse, Result};

struct MyModel;

#[async_trait]
impl Model for MyModel {
    fn name(&self) -> &str { "my-model" }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse> {
        // 调用你自己的后端
        todo!()
    }
}
```

需要 token 级输出时可覆盖 `complete_stream`；默认实现会调用 `complete`，并把最终文本
作为单个 delta 发出，因此老 provider 不需要改代码。核心不强制 reasoning、prompt cache
等厂商差异。

### Tool

工具就是一个异步函数加 JSON Schema：

```rust
use kestrel::{function_tool, ToolOutput, ToolSpec};
use serde_json::json;

let echo = function_tool(
    ToolSpec::new(
        "echo",
        "Echo the supplied text",
        json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        }),
    ),
    |args, _context| async move {
        Ok(ToolOutput::new(args["text"].as_str().unwrap_or_default()))
    },
);
```

工具错误默认为 recoverable：会作为 `Tool error: ...` 反馈给模型。设置
`.fail_on_tool_error(true)` 可以立即中止 run。

### Agent loop

每次模型调用依次执行：

1. 注入基础 system prompt、skills/policy 提示片段和自动召回的记忆。
2. 在超过上下文预算时压缩历史。
3. 调用 `Model::complete_stream`；不支持的模型会回退为单次 `complete`。
4. 追加 assistant 消息；若有 tool call，则依次执行工具并追加结果。
5. 没有 tool call 时返回 `AgentOutput`；否则继续下一 turn。

同一 `Agent` 的并发 run 会串行化。`Agent::clone()` 共享会话；`Agent::fork()` 和
`AgentTemplate::spawn()` 创建隔离会话。

取消使用 `CancellationToken`：

```rust
use kestrel::RunOptions;
use tokio_util::sync::CancellationToken;

let token = CancellationToken::new();
let options = RunOptions::default().with_cancellation(token.clone());
// token.cancel() 可从任意任务触发
let output = agent.run_with("work", options).await?;
```

### 上下文压缩

提供两种内置策略：

- `SlidingWindowCompressor`：确定性保留最近的消息组，无额外模型调用。
- `SummarizingCompressor`：用模型总结旧历史，失败时可回退到滑动窗口。

压缩以原子消息组为单位，不会拆开 assistant tool call 和对应 tool result。有效预算取
显式 `context_window` / `Model::context_window()` 与 compressor 预算中的较小值。

```rust
let compressor = Arc::new(SlidingWindowCompressor::new(80_000));
let agent = Agent::builder(model)
    .context_window(128_000)
    .reserve_output_tokens(8_000)
    .compressor(compressor)
    .build()?;
```

可以注入自定义 `TokenCounter` 以适配目标模型的 tokenizer。默认实现对英文按
4 chars/token、非 ASCII 字符按 1 token 估算。

### 会话与记忆

两个层次刻意分开：

- `SessionStore`：保存完整对话，供下次 run 恢复。
- `Memory`：跨会话的长期事实、偏好和决策，通过检索注入上下文。

```rust
use std::sync::Arc;
use kestrel::{InMemoryMemory, InMemorySessionStore};

let sessions = Arc::new(InMemorySessionStore::new());
let memory = Arc::new(InMemoryMemory::new());

let agent = Agent::builder(model)
    .session_store(sessions)
    .session_id("user-42")
    .memory(memory)
    .with_memory_tools()
    .build()?;
```

接入数据库时实现 `SessionStore` / `Memory` 即可。内置 `InMemoryMemory` 仅用于开发
和测试，不包含 embedding。

### 事件

```rust
use std::sync::Arc;
use kestrel::event_channel;

let (sender, mut events) = event_channel();
let agent = Agent::builder(model)
    .event_sink(Arc::new(sender))
    .build()?;

tokio::spawn(async move {
    while let Some(event) = events.recv().await {
        println!("{event:?}");
    }
});
```

`EventSink::emit` 是同步的，必须快速返回。跨语言 SDK 应转发到线程安全队列或 FFI
callback，不要在 sink 内阻塞。每个事件信封都带 `run_id`、`session_id` 和单调递增的
`sequence`，即使多个 agent 共用一个 channel 也能可靠归组。

### 流式输出

`Agent` 默认启用 streaming。文本和 tool-call 片段通过 `model_delta` 事件流出，
最终完整消息仍由 `AgentOutput` 返回：

```rust
use kestrel::{AgentEventKind, ModelDelta};

while let Some(event) = events.recv().await {
    if let AgentEventKind::ModelDelta {
        delta: ModelDelta::Text { text },
        ..
    } = event.kind {
        print!("{text}");
    }
}
```

OpenAI-compatible provider 使用 SSE，并在本地合并分片的 tool arguments。若服务端不支持
streaming，可关闭：

```rust
let agent = Agent::builder(model)
    .streaming(false)
    .build()?;
```

OpenAI 流式 usage 默认为开启；兼容服务不支持 `stream_options` 时，可对 provider 调用
`.stream_usage(false)`。

## 可选能力

| Feature | 默认 | 内容 |
|---|---:|---|
| `openai` | 是 | OpenAI-compatible Chat Completions provider |
| `todo` | 是 | `TodoList`、`todo_write`、`todo_read` |
| `subagent` | 是 | `SubAgentTool`，每次调用使用隔离会话 |
| `mcp` | 否 | stdio / Streamable HTTP MCP client |
| `skills` | 否 | `SKILL.md` 发现、catalog prompt、`read_skill` |
| `full` | 否 | 启用以上全部 |

只想要零网络依赖的核心：

```toml
kestrel = { path = ".", default-features = false }
```

### Todo：显式开启

Todo 编译进 crate 不等于运行时启用。只有注册工具才会真正暴露给模型：

```rust
let todos = TodoList::new();
let agent = todos.install(Agent::builder(model)).build()?;
```

`TodoList` 是线程安全的共享状态，可以自行持久化/恢复。

### 子 agent

```rust
use std::sync::Arc;
use kestrel::SubAgentTool;

let delegate = SubAgentTool::from_agent(
    "delegate",
    "Delegate a self-contained task to a child agent.",
    &agent,
).with_max_depth(2);
agent.register_tool(Arc::new(delegate))?;
```

子 agent 共享 model、tools、prompt、memory、compressor，但使用独立会话和 run lock。
子 agent 默认不向父级 event sink 暴露内部事件，避免事件流失去上下文。

### MCP

当前支持 **stdio** 和 **Streamable HTTP**，两者都包含 initialize、tools/list 分页和
tools/call：

```rust
use kestrel::mcp::{McpClient, McpServerConfig};

let client = McpClient::connect(
    McpServerConfig::new("npx").args(["-y", "@modelcontextprotocol/server-filesystem", "."]),
).await?;
let tools = client.tools().await?;

let agent = Agent::builder(model).tools(tools).build()?;
// 应用退出前：
// client.close().await?;
```

HTTP endpoint：

```rust
use kestrel::mcp::{McpHttpClient, McpHttpConfig};

let client = McpHttpClient::connect(
    McpHttpConfig::new("https://example.com/mcp")
        .header("Authorization", "Bearer ..."),
).await?;
let tools = client.tools().await?;
```

MCP 工具名会规范化为模型友好的字符集，调用时仍映射回原始远端名称。Streamable HTTP
支持 JSON 和 SSE 响应、session header、协议版本 header 和 DELETE 会话终止；sampling、
roots 及服务端发起的请求仍不在当前实现内。

### Skills

```rust
use kestrel::skills::SkillRegistry;

let skills = SkillRegistry::load_dir("./skills")?;
let builder = skills.install(Agent::builder(model));
let agent = builder.build()?;
```

`install` 会注入 skill catalog，并注册 `read_skill` 工具。当前 frontmatter 支持扁平的
`key: value`，重点字段为 `name` 和 `description`；复杂嵌套 YAML 不在 MVP 内。

## 跨语言 SDK

仓库现在包含独立的 `kestrel-ffi` crate：

- 稳定 C ABI，头文件位于 `kestrel-ffi/include/kestrel.h`
- OpenAI-compatible 或宿主语言自定义 model callback（支持 token delta）
- 宿主语言 tool callback
- 同步事件 callback（包含 `model_delta`）
- 阻塞 run、非阻塞 `KestrelRun`、事件轮询、跨线程取消、Session/Todo JSON 快照
- 通过 builder callback 接入宿主数据库的 SessionStore 与长期 Memory
- 内置 Tokio runtime，不要求宿主语言理解 Rust async

```bash
cargo build -p kestrel-ffi --release
```

配置示例：

```json
{
  "schema_version": 1,
  "system_prompt": "You are a concise assistant.",
  "session_id": "user-42",
  "max_turns": 8,
  "streaming": true,
  "todo": true,
  "memory": { "enabled": true, "tools": true },
  "compression": { "kind": "sliding", "max_tokens": 80000 },
  "skills": { "paths": ["./skills"], "max_depth": 4 },
  "mcp_servers": [
    {
      "url": "https://example.com/mcp",
      "headers": { "Authorization": "Bearer ..." },
      "request_timeout_seconds": 30
    }
  ],
  "openai": {
    "model": "gpt-4.1-mini",
    "api_key": "optional; falls back to OPENAI_API_KEY",
    "base_url": "optional; falls back to OPENAI_BASE_URL"
  }
}
```

自定义模型通过 `kestrel_agent_new_with_model` 注册。callback 收到序列化的
`ModelRequest`，返回序列化的 `ModelResponse`。需要 token 级输出时使用
`kestrel_agent_new_with_streaming_model` 或 Python `stream_model`，生成过程中调用
传入的 `emit` 推送 `ModelDelta`。

`skills` 会在构造时加载并注入 catalog/`read_skill`；`mcp_servers` 会连接 stdio 或
Streamable HTTP MCP 并在构造时发现远端工具。包含 MCP 的构造会阻塞，且不能从 Tokio
runtime 线程调用。

配置契约由独立 `kestrel-contract` crate 提供。各 SDK 使用同一份 schema、
`schema_version`、capability 集合和 `fixtures/` 测试数据；不支持的配置字段或能力会
在构造 Agent 前明确报错，而不是被静默忽略。

`bindings/python/kestrel.py` 是可直接使用的 ctypes 参考绑定：

```bash
cargo build -p kestrel-ffi
PYTHONPATH=bindings/python python3 bindings/python/example.py
PYTHONPATH=bindings/python python3 bindings/python/example_async.py
PYTHONPATH=bindings/python python3 bindings/python/example_persistence.py
PYTHONPATH=bindings/python python3 bindings/python/example_stream_model.py
```

`bindings/go` 是纯 cgo、无第三方 Go 依赖的参考 SDK：

```bash
cargo build -p kestrel-ffi
cd bindings/go
go test ./...
go run ./examples/basic
```

`bindings/node` 是 napi-rs 原生 addon：

```bash
cargo build -p kestrel-node
cd bindings/node
node --test
node example.js
```

其他语言只需要实现 `kestrel.h` 中的函数指针和 JSON 编解码。详细 ABI、内存所有权与
线程约束见 [`kestrel-ffi/README.md`](kestrel-ffi/README.md)。

```text
kestrel             # 核心 Rust library
kestrel-ffi         # C ABI + opaque agent + callback runtime
kestrel-contract    # 版本化配置 schema 与 capability 校验
bindings/python     # ctypes 参考 SDK
bindings/go         # cgo 参考 SDK
bindings/node       # napi-rs 原生 addon
```

## 当前边界

- OpenAI provider 和自定义 streaming model callback 都支持 token/tool-call 增量。
- 单一模型调用内不并发执行工具，保证结果顺序和错误语义简单可预测。
- OpenAI provider 使用 Chat Completions 兼容协议，不覆盖 Responses API。
- MCP 支持 stdio 与 Streamable HTTP；sampling/roots 和 server-initiated requests 尚未支持。
- FFI 同时提供同步 `run` 和异步 `start_run`；异步 handle 的 `poll/wait` 本身是同步
  C 调用，语言 SDK 应放到 worker 或 executor。
- FFI 的 tool callback 是同步回调；取消无法中断已经进入宿主 callback 的调用，但 run 会等
  在途 callback 返回后才报告取消，确保 `user_data` 生命周期安全。
- Node addon 目前未暴露 MCP 配置和显式 run handle。
- 会话压缩会更新持久化历史；如需保留完整审计日志，应在 SessionStore 外单独存 event/audit。
- 摘要压缩的额外模型调用，以及子 agent 的 token 用量，不计入父级 `AgentOutput.usage`。

这些边界都是可扩展点，而不是核心循环中的硬编码假设。

## 验证

```bash
cargo fmt --all -- --check
bash scripts/check-line-limits.sh
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo test --workspace --all-features
cargo check -p kestrel --no-default-features
cc -fsyntax-only -x c -include kestrel-ffi/include/kestrel.h /dev/null
cargo build -p kestrel-ffi
PYTHONPATH=bindings/python python3 bindings/python/example.py
PYTHONPATH=bindings/python python3 bindings/python/example_async.py
PYTHONPATH=bindings/python python3 bindings/python/example_persistence.py
PYTHONPATH=bindings/python python3 bindings/python/example_stream_model.py
cd bindings/python && python3 -m unittest test_contract.py
cd bindings/go && go vet ./... && go test ./...
cargo build -p kestrel-node
cd bindings/node && node --test
```

## 目录

```text
src/
  agent/        # config、builder、run loop
  context/      # token 估算、滑动窗口、摘要压缩
  memory/       # 长期记忆 trait、内存实现、工具
  session.rs    # 会话与持久化
  model.rs      # LLM 抽象
  tool.rs       # Tool trait、function_tool、registry
  event.rs      # 可序列化事件与 sink
  openai.rs     # 可选 OpenAI-compatible provider
  todo.rs       # 可选 Todo
  subagent.rs   # 可选子 agent 工具
  mcp/          # 可选 stdio / Streamable HTTP MCP
  skills/       # 可选 SKILL.md 加载
kestrel-ffi/
  include/      # C header
  src/          # opaque agent、callback、model/tool 适配
kestrel-contract/ # 共享配置 schema、capability、golden fixtures
kestrel-node/    # napi-rs Rust addon
bindings/python/
  kestrel.py    # ctypes SDK
  example.py    # 离线 callback model + tool 示例
  example_async.py # 非阻塞 run + 事件轮询示例
  example_persistence.py # 宿主 SessionStore + Memory 示例
  example_stream_model.py # 自定义模型 token 流示例
bindings/go/
  types.go       # Message、Tool、Memory 等类型
  agent.go       # Agent/Run API
  callbacks.go   # cgo bridge 与 callback registry
  examples/      # 可运行示例
bindings/node/
  index.js       # native addon loader
  index.d.ts     # TypeScript 定义
  example.js     # Node 示例
```
