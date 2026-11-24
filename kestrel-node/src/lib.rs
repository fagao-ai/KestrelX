use std::{
    sync::{Arc, RwLock},
    time::Duration,
};

use async_trait::async_trait;
use kestrel::skills::SkillRegistry;
use kestrel::{
    Agent as CoreAgent, AgentConfig, AgentEvent, ContextCompressor, EventSink, InMemoryMemory,
    MemoryOptions, Model, ModelRequest, ModelResponse, OpenAiCompatible, Result as CoreResult,
    SlidingWindowCompressor, SubAgentTool, SummarizingCompressor, TodoList, Tool, ToolChoice,
    ToolContext, ToolOutput, ToolSpec,
};
use napi::{
    Error, Result, Status,
    bindgen_prelude::{FnArgs, Function, Promise},
    threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode, UnknownReturnValue},
};
use napi_derive::napi;
use serde::Deserialize;
use serde_json::Value;

type ModelFn = ThreadsafeFunction<Value, Promise<Value>, Value, Status, false>;
type ToolFn = ThreadsafeFunction<
    FnArgs<(String, Value)>,
    Promise<Value>,
    FnArgs<(String, Value)>,
    Status,
    false,
>;
type EventFn = ThreadsafeFunction<Value, UnknownReturnValue, Value, Status, false>;

fn node_error(error: impl std::fmt::Display) -> Error {
    Error::new(Status::GenericFailure, error.to_string())
}

fn build_model_callback(
    callback: Function<'static, Value, Promise<Value>>,
) -> Result<Arc<ModelFn>> {
    let callback: ModelFn = callback
        .build_threadsafe_function::<Value>()
        .callee_handled::<false>()
        .build()?;
    Ok(Arc::new(callback))
}

fn build_tool_callback(
    callback: Function<'static, FnArgs<(String, Value)>, Promise<Value>>,
) -> Result<Arc<ToolFn>> {
    let callback: ToolFn = callback
        .build_threadsafe_function::<FnArgs<(String, Value)>>()
        .callee_handled::<false>()
        .build()?;
    Ok(Arc::new(callback))
}

fn build_event_callback(
    callback: Function<'static, Value, UnknownReturnValue>,
) -> Result<Arc<EventFn>> {
    let callback: EventFn = callback
        .build_threadsafe_function::<Value>()
        .callee_handled::<false>()
        .build()?;
    Ok(Arc::new(callback))
}

struct JsModel {
    name: String,
    context_window: Option<usize>,
    callback: Arc<ModelFn>,
}

#[async_trait]
impl Model for JsModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn context_window(&self) -> Option<usize> {
        self.context_window
    }

    async fn complete(&self, request: ModelRequest) -> CoreResult<ModelResponse> {
        let request_value = serde_json::to_value(request)
            .map_err(|error| kestrel::Error::Model(error.to_string()))?;
        let promise = self
            .callback
            .call_async(request_value)
            .await
            .map_err(|error| kestrel::Error::Model(error.to_string()))?;
        let response_value = promise
            .await
            .map_err(|error| kestrel::Error::Model(error.to_string()))?;
        serde_json::from_value(response_value).map_err(|error| {
            kestrel::Error::InvalidModelResponse(format!(
                "JavaScript model returned invalid response: {error}"
            ))
        })
    }
}

struct JsTool {
    spec: ToolSpec,
    callback: Arc<ToolFn>,
}

#[async_trait]
impl Tool for JsTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, arguments: Value, context: &ToolContext) -> CoreResult<ToolOutput> {
        let input = FnArgs::from((context.call_id.clone(), arguments));
        let promise = self
            .callback
            .call_async(input)
            .await
            .map_err(|error| kestrel::Error::tool(&self.spec.name, error.to_string()))?;
        let value = promise
            .await
            .map_err(|error| kestrel::Error::tool(&self.spec.name, error.to_string()))?;
        match serde_json::from_value::<ToolOutput>(value.clone()) {
            Ok(output) => Ok(output),
            Err(_) => match value.as_str() {
                Some(text) => Ok(ToolOutput::new(text)),
                None => Ok(ToolOutput::error(
                    "JavaScript tool must return a string or ToolOutput object",
                )),
            },
        }
    }
}

#[derive(Clone, Default)]
struct EventSlot {
    handler: Arc<RwLock<Option<Arc<EventFn>>>>,
}

impl EventSlot {
    fn set(&self, handler: Option<Function<'static, Value, UnknownReturnValue>>) -> Result<()> {
        *self
            .handler
            .write()
            .unwrap_or_else(|error| error.into_inner()) =
            handler.map(build_event_callback).transpose()?;
        Ok(())
    }
}

struct JsEventSink {
    slot: EventSlot,
}

impl EventSink for JsEventSink {
    fn emit(&self, event: AgentEvent) {
        let Ok(value) = serde_json::to_value(event) else {
            return;
        };
        let handler = self
            .slot
            .handler
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .clone();
        if let Some(handler) = handler {
            let _ = handler.call(value, ThreadsafeFunctionCallMode::NonBlocking);
        }
    }
}

/// JavaScript agent handle.
#[napi]
pub struct Agent {
    inner: Option<CoreAgent>,
    events: EventSlot,
    todos: Option<TodoList>,
}

/// Creates a Kestrel agent.
#[napi]
pub fn create_agent(
    config: Value,
    model: Option<Function<'static, Value, Promise<Value>>>,
    event_handler: Option<Function<'static, Value, UnknownReturnValue>>,
) -> Result<Agent> {
    let config: NodeConfig = serde_json::from_value(config).map_err(node_error)?;
    let events = EventSlot::default();
    events.set(event_handler)?;
    let model = model.map(build_model_callback).transpose()?;
    config.build(model, events).map_err(node_error)
}

#[napi]
impl Agent {
    fn core(&self) -> Result<&CoreAgent> {
        self.inner
            .as_ref()
            .ok_or_else(|| Error::new(Status::GenericFailure, "agent is closed"))
    }

    /// Runs one user prompt.
    #[napi]
    pub async fn run(&self, prompt: String) -> Result<Value> {
        let inner = self.core()?.clone();
        let output = inner.run(prompt).await.map_err(node_error)?;
        serde_json::to_value(output).map_err(node_error)
    }

    /// Returns a session snapshot.
    #[napi]
    pub async fn session(&self) -> Result<Value> {
        let inner = self.core()?.clone();
        let session = inner.session().await;
        serde_json::to_value(session).map_err(node_error)
    }

    /// Returns todo items, or null when todos are disabled.
    #[napi]
    pub fn todos(&self) -> Result<Option<Value>> {
        self.todos
            .as_ref()
            .map(|todos| serde_json::to_value(todos.snapshot()).map_err(node_error))
            .transpose()
    }

    /// Clears history while retaining the session ID.
    #[napi]
    pub async fn reset(&self) -> Result<()> {
        self.core()?.clone().reset().await.map_err(node_error)
    }

    /// Replaces the event handler.
    #[napi]
    pub fn set_event_handler(
        &self,
        handler: Option<Function<'static, Value, UnknownReturnValue>>,
    ) -> Result<()> {
        self.events.set(handler)
    }

    /// Registers one asynchronous JavaScript tool.
    #[napi]
    pub fn register_tool(
        &self,
        spec: Value,
        callback: Function<'static, FnArgs<(String, Value)>, Promise<Value>>,
    ) -> Result<()> {
        let spec: ToolSpec = serde_json::from_value(spec).map_err(node_error)?;
        let callback = build_tool_callback(callback)?;
        self.core()?
            .register_tool(Arc::new(JsTool { spec, callback }))
            .map_err(node_error)
    }

    /// Releases native resources. All runs must be awaited first.
    #[napi]
    pub fn close(&mut self) {
        self.inner = None;
        self.todos = None;
        self.events.set(None).ok();
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
struct NodeConfig {
    system_prompt: Option<String>,
    prompt_sections: Vec<String>,
    session_id: Option<String>,
    max_turns: usize,
    temperature: Option<f32>,
    max_output_tokens: Option<u32>,
    context_window: Option<usize>,
    reserve_output_tokens: usize,
    streaming: bool,
    fail_on_tool_error: bool,
    tool_choice: Option<ToolChoice>,
    todo: bool,
    memory: MemorySettings,
    compression: Option<CompressionSettings>,
    subagent: Option<SubagentSettings>,
    skills: Option<SkillsSettings>,
    openai: OpenAiSettings,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            system_prompt: None,
            prompt_sections: Vec::new(),
            session_id: None,
            max_turns: 16,
            temperature: None,
            max_output_tokens: None,
            context_window: None,
            reserve_output_tokens: 4_096,
            streaming: true,
            fail_on_tool_error: false,
            tool_choice: None,
            todo: false,
            memory: MemorySettings::default(),
            compression: None,
            subagent: None,
            skills: None,
            openai: OpenAiSettings::default(),
        }
    }
}

impl NodeConfig {
    fn build(self, model: Option<Arc<ModelFn>>, events: EventSlot) -> CoreResult<Agent> {
        let model = match model {
            Some(callback) => Arc::new(JsModel {
                name: if self.openai.model.is_empty() {
                    "js-model".to_owned()
                } else {
                    self.openai.model.clone()
                },
                context_window: self.context_window.or(self.openai.context_window),
                callback,
            }) as Arc<dyn Model>,
            None => Arc::new(self.openai.build()?) as Arc<dyn Model>,
        };

        let mut config = AgentConfig {
            max_turns: self.max_turns,
            temperature: self.temperature,
            max_output_tokens: self.max_output_tokens,
            context_window: self.context_window,
            reserve_output_tokens: self.reserve_output_tokens,
            streaming: self.streaming,
            fail_on_tool_error: self.fail_on_tool_error,
            ..Default::default()
        };
        if let Some(tool_choice) = self.tool_choice {
            config.tool_choice = tool_choice;
        }

        let mut builder = CoreAgent::builder(model.clone())
            .config(config)
            .event_sink(Arc::new(JsEventSink {
                slot: events.clone(),
            }));
        if let Some(system_prompt) = self.system_prompt {
            builder = builder.system_prompt(system_prompt);
        }
        for section in self.prompt_sections {
            builder = builder.prompt_section(section);
        }
        if let Some(session_id) = self.session_id {
            builder = builder.session_id(session_id);
        }
        if let Some(skills) = self.skills {
            let depth = skills.max_depth();
            let mut registry = SkillRegistry::default();
            for path in skills.paths {
                let loaded = SkillRegistry::load_dir_with_depth(&path, depth)?;
                for skill in loaded.skills() {
                    registry.insert((*skill).clone())?;
                }
            }
            builder = registry.install(builder);
        }

        let mut todos = None;
        if self.todo {
            let list = TodoList::new();
            builder = list.install(builder);
            todos = Some(list);
        }

        if self.memory.enabled {
            builder = builder
                .memory(Arc::new(InMemoryMemory::new()))
                .memory_options(MemoryOptions {
                    auto_recall: self.memory.auto_recall,
                    recall_limit: self.memory.recall_limit,
                    max_context_tokens: self.memory.max_context_tokens,
                });
            if self.memory.tools {
                builder = builder.with_memory_tools();
            }
        } else if self.memory.tools {
            return Err(kestrel::Error::Configuration(
                "memory.tools requires memory.enabled".to_owned(),
            ));
        }

        if let Some(compression) = self.compression {
            if compression.max_tokens == 0 {
                return Err(kestrel::Error::Configuration(
                    "compression.max_tokens must be greater than zero".to_owned(),
                ));
            }
            let compressor: Arc<dyn ContextCompressor> = match compression.kind {
                CompressionKind::Sliding => {
                    Arc::new(SlidingWindowCompressor::new(compression.max_tokens))
                }
                CompressionKind::Summarizing => {
                    let mut compressor = SummarizingCompressor::new(model, compression.max_tokens);
                    if let Some(tokens) = compression.summary_tokens {
                        compressor = compressor.with_summary_tokens(tokens);
                    }
                    Arc::new(compressor)
                }
            };
            builder = builder.compressor(compressor);
        }

        let inner = builder.build()?;
        if let Some(subagent) = self.subagent {
            let tool = SubAgentTool::from_agent(subagent.name, subagent.description, &inner)
                .with_max_depth(subagent.max_depth);
            inner.register_tool(Arc::new(tool))?;
        }
        Ok(Agent {
            inner: Some(inner),
            events,
            todos,
        })
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct OpenAiSettings {
    model: String,
    api_key: Option<String>,
    base_url: Option<String>,
    context_window: Option<usize>,
    timeout_seconds: Option<u64>,
    stream_usage: Option<bool>,
}

impl OpenAiSettings {
    fn build(&self) -> CoreResult<OpenAiCompatible> {
        if self.model.trim().is_empty() {
            return Err(kestrel::Error::Configuration(
                "openai.model is required when no JavaScript model is supplied".to_owned(),
            ));
        }
        let mut builder = OpenAiCompatible::builder(&self.model);
        if let Some(api_key) = &self.api_key {
            builder = builder.api_key(api_key);
        } else if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            builder = builder.api_key(api_key);
        }
        if let Some(base_url) = &self.base_url {
            builder = builder.base_url(base_url);
        } else if let Ok(base_url) = std::env::var("OPENAI_BASE_URL") {
            builder = builder.base_url(base_url);
        }
        if let Some(tokens) = self.context_window {
            builder = builder.context_window(tokens);
        }
        if let Some(seconds) = self.timeout_seconds {
            if seconds == 0 {
                return Err(kestrel::Error::Configuration(
                    "openai.timeout_seconds must be greater than zero".to_owned(),
                ));
            }
            builder = builder.timeout(Duration::from_secs(seconds));
        }
        if let Some(stream_usage) = self.stream_usage {
            builder = builder.stream_usage(stream_usage);
        }
        builder.build()
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct MemorySettings {
    enabled: bool,
    tools: bool,
    auto_recall: bool,
    recall_limit: usize,
    max_context_tokens: usize,
}

impl Default for MemorySettings {
    fn default() -> Self {
        Self {
            enabled: false,
            tools: false,
            auto_recall: true,
            recall_limit: 5,
            max_context_tokens: 2_000,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct SkillsSettings {
    paths: Vec<String>,
    max_depth: usize,
}

impl SkillsSettings {
    fn max_depth(&self) -> usize {
        if self.max_depth == 0 {
            4
        } else {
            self.max_depth
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct CompressionSettings {
    #[serde(default)]
    kind: CompressionKind,
    max_tokens: usize,
    summary_tokens: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CompressionKind {
    #[default]
    Sliding,
    Summarizing,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct SubagentSettings {
    name: String,
    description: String,
    max_depth: usize,
}

impl Default for SubagentSettings {
    fn default() -> Self {
        Self {
            name: "delegate".to_owned(),
            description: "Delegate a self-contained task to a child agent.".to_owned(),
            max_depth: 2,
        }
    }
}
