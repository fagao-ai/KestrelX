mod builder;
mod runtime;

use std::sync::{Arc, atomic::AtomicBool};

use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;

use crate::{
    ContextCompressor, EventSink, FinishReason, Memory, MemoryOptions, Message, Model, Session,
    SessionStore, TokenCounter, ToolChoice, ToolRegistry, Usage,
};

pub use builder::AgentBuilder;

/// Runtime tuning shared by all runs of an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentConfig {
    /// Maximum number of model calls in one run.
    pub max_turns: usize,
    /// Explicit model context window. Overrides [`Model::context_window`].
    pub context_window: Option<usize>,
    /// Tokens reserved for the next model output.
    pub reserve_output_tokens: usize,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum output tokens per model call.
    pub max_output_tokens: Option<u32>,
    /// Tool selection mode sent to the model.
    pub tool_choice: ToolChoice,
    /// Abort instead of feeding recoverable tool errors back to the model.
    pub fail_on_tool_error: bool,
    /// Request incremental model output. Models without native streaming still
    /// emit their final text as one delta.
    pub streaming: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_turns: 16,
            context_window: None,
            reserve_output_tokens: 4_096,
            temperature: None,
            max_output_tokens: None,
            tool_choice: ToolChoice::Auto,
            fail_on_tool_error: false,
            streaming: true,
        }
    }
}

/// Per-run options.
#[derive(Clone, Default)]
pub struct RunOptions {
    /// Cooperative cancellation token.
    pub cancellation: Option<CancellationToken>,
    /// Overrides the agent's default event sink for this run.
    pub event_sink: Option<Arc<dyn EventSink>>,
}

impl std::fmt::Debug for RunOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RunOptions")
            .field("cancellation", &self.cancellation)
            .field("has_event_sink", &self.event_sink.is_some())
            .finish()
    }
}

impl RunOptions {
    /// Sets the cancellation token.
    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = Some(cancellation);
        self
    }

    /// Sets an event sink for this run.
    pub fn with_event_sink(mut self, sink: Arc<dyn EventSink>) -> Self {
        self.event_sink = Some(sink);
        self
    }
}

/// Final result of one agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOutput {
    /// Unique run identifier.
    pub run_id: String,
    /// Conversation identifier.
    pub session_id: String,
    /// Final assistant message.
    pub message: Message,
    /// Concatenated final assistant text.
    pub text: String,
    /// Number of model calls used.
    pub turns: usize,
    /// Accumulated token usage.
    pub usage: Usage,
    /// Finish reason from the final model call.
    pub finish_reason: FinishReason,
}

pub(crate) struct AgentShared {
    pub model: Arc<dyn Model>,
    pub config: AgentConfig,
    pub system_prompt: Option<String>,
    pub prompt_sections: Vec<String>,
    pub tools: ToolRegistry,
    pub compressor: Option<Arc<dyn ContextCompressor>>,
    pub token_counter: Arc<dyn TokenCounter>,
    pub memory: Option<Arc<dyn Memory>>,
    pub memory_options: MemoryOptions,
}

pub(crate) struct AgentState {
    pub session: RwLock<Session>,
    pub session_store: Option<Arc<dyn SessionStore>>,
    pub event_sink: Option<Arc<dyn EventSink>>,
    pub loaded: AtomicBool,
    pub depth: usize,
    pub run_lock: Mutex<()>,
}

/// A reusable, session-independent agent blueprint.
///
/// Templates keep the model, tools, prompts, memory, and compression
/// configuration, but not conversation state or event subscribers. They are
/// useful for sub-agent adapters and any application that needs isolated runs
/// from a reusable configuration.
#[derive(Clone)]
pub struct AgentTemplate {
    shared: Arc<AgentShared>,
    base_depth: usize,
}

impl AgentTemplate {
    /// Returns the model name.
    pub fn model_name(&self) -> &str {
        self.shared.model.name()
    }

    /// Spawns a fresh agent with an empty session.
    pub fn spawn(&self) -> Agent {
        self.spawn_at_depth(self.base_depth.saturating_add(1))
    }

    pub(crate) fn spawn_at_depth(&self, depth: usize) -> Agent {
        Agent {
            shared: self.shared.clone(),
            state: Arc::new(AgentState {
                session: RwLock::new(Session::new()),
                session_store: None,
                event_sink: None,
                loaded: AtomicBool::new(true),
                depth,
                run_lock: Mutex::new(()),
            }),
        }
    }
}

/// A reusable agent with a shared conversation.
///
/// Cloning an agent shares its session and run lock. Use [`Agent::fork`] or
/// [`Agent::template`] when isolated conversation state is needed.
#[derive(Clone)]
pub struct Agent {
    shared: Arc<AgentShared>,
    state: Arc<AgentState>,
}

impl Agent {
    /// Creates a builder around a model implementation.
    pub fn builder<M>(model: M) -> AgentBuilder
    where
        M: Model + 'static,
    {
        AgentBuilder::new(Arc::new(model))
    }

    /// Returns the active configuration.
    pub fn config(&self) -> &AgentConfig {
        &self.shared.config
    }

    /// Returns the model name.
    pub fn model_name(&self) -> &str {
        self.shared.model.name()
    }

    /// Returns the live tool registry.
    pub fn tools(&self) -> &ToolRegistry {
        &self.shared.tools
    }

    /// Registers a tool at runtime.
    pub fn register_tool(&self, tool: Arc<dyn crate::Tool>) -> crate::Result<()> {
        self.shared.tools.register(tool)
    }

    /// Returns a snapshot of the current session.
    pub async fn session(&self) -> Session {
        self.state.session.read().await.clone()
    }

    /// Returns the current session ID.
    pub async fn session_id(&self) -> String {
        self.state.session.read().await.id.clone()
    }

    /// Returns a snapshot of the current history.
    pub async fn messages(&self) -> Vec<Message> {
        self.state.session.read().await.messages.clone()
    }

    /// Creates an isolated blueprint from this agent.
    pub fn template(&self) -> AgentTemplate {
        AgentTemplate {
            shared: self.shared.clone(),
            base_depth: self.state.depth,
        }
    }

    /// Spawns an isolated agent with fresh conversation state.
    pub fn fork(&self) -> Agent {
        self.template().spawn()
    }

    pub(crate) fn from_parts(shared: Arc<AgentShared>, state: AgentState) -> Self {
        Self {
            shared,
            state: Arc::new(state),
        }
    }
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Agent")
            .field("model", &self.shared.model.name())
            .field("tools", &self.shared.tools.specs())
            .finish_non_exhaustive()
    }
}
