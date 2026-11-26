use std::sync::{Arc, atomic::AtomicBool};

use tokio::sync::{Mutex, RwLock};

use crate::ToolConcurrency;
use crate::{
    Agent, AgentConfig, ContextCompressor, EventSink, HeuristicTokenCounter, Memory, MemoryOptions,
    Model, Session, SessionStore, TokenCounter, Tool, ToolRegistry, memory_tools,
};

use super::{AgentShared, AgentState};

/// Builder for [`Agent`].
pub struct AgentBuilder {
    model: Arc<dyn Model>,
    config: AgentConfig,
    system_prompt: Option<String>,
    prompt_sections: Vec<String>,
    tools: Vec<Arc<dyn Tool>>,
    registry: Option<ToolRegistry>,
    session: Option<Session>,
    session_id: Option<String>,
    session_store: Option<Arc<dyn SessionStore>>,
    compressor: Option<Arc<dyn ContextCompressor>>,
    token_counter: Arc<dyn TokenCounter>,
    memory: Option<Arc<dyn Memory>>,
    memory_options: MemoryOptions,
    memory_tools_enabled: bool,
    event_sink: Option<Arc<dyn EventSink>>,
}

impl AgentBuilder {
    pub(crate) fn new(model: Arc<dyn Model>) -> Self {
        Self {
            model,
            config: AgentConfig::default(),
            system_prompt: None,
            prompt_sections: Vec::new(),
            tools: Vec::new(),
            registry: None,
            session: None,
            session_id: None,
            session_store: None,
            compressor: None,
            token_counter: Arc::new(HeuristicTokenCounter),
            memory: None,
            memory_options: MemoryOptions::default(),
            memory_tools_enabled: false,
            event_sink: None,
        }
    }

    /// Sets the base system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Appends a prompt section after the base system prompt.
    ///
    /// This is the intended integration point for skills, policies, and other
    /// prompt-based extensions.
    pub fn prompt_section(mut self, section: impl Into<String>) -> Self {
        self.prompt_sections.push(section.into());
        self
    }

    /// Adds one tool.
    pub fn tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Adds multiple tools.
    pub fn tools(mut self, tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Uses an existing dynamic registry.
    pub fn tool_registry(mut self, registry: ToolRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Supplements the base prompt with a complete configuration.
    pub fn config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the maximum number of model turns.
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.config.max_turns = max_turns;
        self
    }

    /// Sets an explicit model context window.
    pub fn context_window(mut self, tokens: usize) -> Self {
        self.config.context_window = Some(tokens);
        self
    }

    /// Sets output tokens reserved inside the context window.
    pub fn reserve_output_tokens(mut self, tokens: usize) -> Self {
        self.config.reserve_output_tokens = tokens;
        self
    }

    /// Sets model temperature.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    /// Sets maximum output tokens per model call.
    pub fn max_output_tokens(mut self, tokens: u32) -> Self {
        self.config.max_output_tokens = Some(tokens);
        self
    }

    /// Sets the tool selection policy.
    pub fn tool_choice(mut self, tool_choice: crate::ToolChoice) -> Self {
        self.config.tool_choice = tool_choice;
        self
    }

    /// Aborts on tool errors instead of returning them to the model.
    pub fn fail_on_tool_error(mut self, enabled: bool) -> Self {
        self.config.fail_on_tool_error = enabled;
        self
    }

    /// Enables or disables incremental model deltas.
    pub fn streaming(mut self, enabled: bool) -> Self {
        self.config.streaming = enabled;
        self
    }

    /// Sets a per-tool execution timeout in milliseconds.
    pub fn tool_timeout_ms(mut self, timeout_ms: Option<u64>) -> Self {
        self.config.tool_timeout_ms = timeout_ms;
        self
    }

    /// Limits retained characters from one tool output.
    pub fn max_tool_output_chars(mut self, max_chars: Option<usize>) -> Self {
        self.config.max_tool_output_chars = max_chars;
        self
    }

    /// Sets whether tool calls from one assistant message run concurrently.
    pub fn tool_concurrency(mut self, concurrency: ToolConcurrency) -> Self {
        self.config.tool_concurrency = concurrency;
        self
    }

    /// Uses an existing session.
    pub fn session(mut self, session: Session) -> Self {
        self.session = Some(session);
        self
    }

    /// Sets a stable session ID. When a store is configured, the ID is loaded
    /// lazily on the first run.
    pub fn session_id(mut self, id: impl Into<String>) -> Self {
        self.session_id = Some(id.into());
        self
    }

    /// Enables automatic session load/save.
    pub fn session_store(mut self, store: Arc<dyn SessionStore>) -> Self {
        self.session_store = Some(store);
        self
    }

    /// Installs a context compressor.
    pub fn compressor(mut self, compressor: Arc<dyn ContextCompressor>) -> Self {
        self.compressor = Some(compressor);
        self
    }

    /// Installs a custom token estimator.
    pub fn token_counter(mut self, counter: Arc<dyn TokenCounter>) -> Self {
        self.token_counter = counter;
        self
    }

    /// Attaches long-term memory and enables automatic recall.
    pub fn memory(mut self, memory: Arc<dyn Memory>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Tunes long-term memory behavior.
    pub fn memory_options(mut self, options: MemoryOptions) -> Self {
        self.memory_options = options;
        self
    }

    /// Registers `memory_remember` and `memory_recall` tools.
    pub fn with_memory_tools(mut self) -> Self {
        self.memory_tools_enabled = true;
        self
    }

    /// Sets a default event sink.
    pub fn event_sink(mut self, sink: Arc<dyn EventSink>) -> Self {
        self.event_sink = Some(sink);
        self
    }

    /// Builds the agent.
    pub fn build(self) -> crate::Result<Agent> {
        if self.config.max_turns == 0 {
            return Err(crate::Error::Configuration(
                "max_turns must be at least 1".to_owned(),
            ));
        }
        if let Some(window) = self.config.context_window {
            if window == 0 {
                return Err(crate::Error::Configuration(
                    "context_window must be greater than zero".to_owned(),
                ));
            }
            if self.config.reserve_output_tokens >= window {
                return Err(crate::Error::Configuration(
                    "reserve_output_tokens must be smaller than context_window".to_owned(),
                ));
            }
        }
        if self.config.max_output_tokens == Some(0) {
            return Err(crate::Error::Configuration(
                "max_output_tokens must be greater than zero".to_owned(),
            ));
        }
        if self.config.tool_timeout_ms == Some(0) {
            return Err(crate::Error::Configuration(
                "tool_timeout_ms must be greater than zero".to_owned(),
            ));
        }
        if self.config.max_tool_output_chars == Some(0) {
            return Err(crate::Error::Configuration(
                "max_tool_output_chars must be greater than zero".to_owned(),
            ));
        }
        if self.memory_tools_enabled && self.memory.is_none() {
            return Err(crate::Error::Configuration(
                "memory tools require a Memory implementation".to_owned(),
            ));
        }
        if let Some(id) = &self.session_id
            && id.trim().is_empty()
        {
            return Err(crate::Error::Configuration(
                "session id cannot be empty".to_owned(),
            ));
        }
        if let (Some(session), Some(id)) = (&self.session, &self.session_id)
            && session.id != *id
        {
            return Err(crate::Error::Configuration(format!(
                "session id `{}` does not match configured session_id `{id}`",
                session.id
            )));
        }
        if let Some(session) = &self.session
            && session.id.trim().is_empty()
        {
            return Err(crate::Error::Configuration(
                "session id cannot be empty".to_owned(),
            ));
        }

        let registry = self.registry.unwrap_or_default();
        let mut tools = self.tools;
        if self.memory_tools_enabled {
            let memory = self.memory.clone().ok_or_else(|| {
                crate::Error::Configuration(
                    "memory tools require a Memory implementation".to_owned(),
                )
            })?;
            tools.extend(memory_tools(memory));
        }
        registry.register_all(tools)?;

        let load_from_store =
            self.session.is_none() && self.session_store.is_some() && self.session_id.is_some();
        let session = match self.session {
            Some(session) => session,
            None => self.session_id.map(Session::with_id).unwrap_or_default(),
        };

        let shared = Arc::new(AgentShared {
            model: self.model,
            config: self.config,
            system_prompt: self.system_prompt,
            prompt_sections: self.prompt_sections,
            tools: registry,
            compressor: self.compressor,
            token_counter: self.token_counter,
            memory: self.memory,
            memory_options: self.memory_options,
        });

        Ok(Agent::from_parts(
            shared,
            AgentState {
                session: RwLock::new(session),
                session_store: self.session_store,
                event_sink: self.event_sink,
                loaded: AtomicBool::new(!load_from_store),
                depth: 0,
                run_lock: Mutex::new(()),
            },
        ))
    }
}

impl std::fmt::Debug for AgentBuilder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AgentBuilder")
            .field("model", &self.model.name())
            .field("config", &self.config)
            .field("tool_count", &self.tools.len())
            .field("has_memory", &self.memory.is_some())
            .field("has_compressor", &self.compressor.is_some())
            .finish()
    }
}
