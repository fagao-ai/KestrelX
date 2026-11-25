//! Kestrel is a small, scenario-agnostic agent runtime.
//!
//! The core deliberately knows nothing about coding, office work, files, or
//! shells. Applications provide a [`Model`], register [`Tool`]s, and decide
//! which optional capabilities to install.
//!
//! # Quick start
//!
//! ```no_run
//! use std::sync::Arc;
//! use kestrel::{Agent, Model, Result};
//!
//! # async fn example(model: Arc<dyn Model>) -> Result<()> {
//! let agent = Agent::builder(model)
//!     .system_prompt("You are a concise assistant.")
//!     .max_turns(8)
//!     .build()?;
//!
//! let output = agent.run("Hello").await?;
//! println!("{}", output.text);
//! # Ok(())
//! # }
//! ```
//!
//! Optional integrations are feature-gated: `openai` (default), `subagent`
//! (default), `mcp`, `skills`, and `todo`.
//!
//! Streaming providers can override [`Model::complete_stream`]. The agent
//! forwards text and tool-call fragments through
//! [`AgentEventKind::ModelDelta`]. Models without native streaming still work
//! through the default single-chunk fallback.

mod agent;
mod context;
mod error;
mod event;
mod memory;
mod message;
mod model;
mod session;
mod tool;
mod util;

#[cfg(feature = "mcp")]
pub mod mcp;
#[cfg(feature = "openai")]
mod openai;
#[cfg(feature = "skills")]
pub mod skills;
#[cfg(feature = "subagent")]
mod subagent;
#[cfg(feature = "todo")]
mod todo;

pub use agent::{Agent, AgentBuilder, AgentConfig, AgentOutput, AgentTemplate, RunOptions};
pub use context::{
    Compression, CompressionRequest, ContextCompressor, HeuristicTokenCounter,
    SlidingWindowCompressor, SummarizingCompressor, TokenCounter,
};
pub use error::{Error, Result};
pub use event::{AgentEvent, AgentEventKind, EventSink, event_channel};
pub use memory::{
    InMemoryMemory, Memory, MemoryEntry, MemoryOptions, MemoryQuery, RecallTool, RememberTool,
    memory_tools,
};
pub use message::{ContentPart, FinishReason, Message, Role, ToolCall, Usage};
pub use model::{Model, ModelDelta, ModelRequest, ModelResponse, ToolChoice};
pub use session::{
    InMemorySessionStore, RunCheckpoint, RunPhase, SESSION_SCHEMA_VERSION, Session, SessionStore,
};
pub use tool::{Tool, ToolContext, ToolOutput, ToolRegistry, ToolSpec, function_tool};

#[cfg(feature = "openai")]
pub use openai::{OpenAiCompatible, OpenAiCompatibleBuilder};
#[cfg(feature = "subagent")]
pub use subagent::SubAgentTool;
#[cfg(feature = "todo")]
pub use todo::{TodoItem, TodoList, TodoReadTool, TodoStatus, TodoWriteTool};
