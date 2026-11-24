use std::{
    sync::{Arc, RwLock},
    time::Duration,
};

use async_trait::async_trait;
use kestrel::skills::SkillRegistry;
use kestrel::{
    Agent as CoreAgent, AgentEvent, ContextCompressor, EventSink, InMemoryMemory, Model,
    ModelRequest, ModelResponse, OpenAiCompatible, Result as CoreResult, SlidingWindowCompressor,
    SubAgentTool, SummarizingCompressor, TodoList, Tool, ToolContext, ToolOutput, ToolSpec,
};
use kestrel_contract::{
    AgentConfig as WireConfig, Capability, CapabilitySet, CompressionKind, OpenAiSettings,
    SCHEMA_VERSION,
};
use napi::{
    Error, Result, Status,
    bindgen_prelude::{FnArgs, Function, Promise},
    threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode, UnknownReturnValue},
};
use napi_derive::napi;
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

fn node_capabilities() -> CapabilitySet {
    CapabilitySet::new([
        Capability::OpenAi,
        Capability::CustomModel,
        Capability::Streaming,
        Capability::Todo,
        Capability::Memory,
        Capability::CompressionSliding,
        Capability::CompressionSummarizing,
        Capability::Subagent,
        Capability::Skills,
        Capability::HostTools,
    ])
}

/// Returns the shared configuration schema version.
#[napi]
pub fn contract_version() -> u32 {
    SCHEMA_VERSION
}

/// Returns capabilities implemented by this Node addon.
#[napi]
pub fn capabilities() -> Vec<&'static str> {
    node_capabilities().names()
}

/// Validates config JSON against the shared contract and Node capabilities.
#[napi]
pub fn validate_config(config: Value, has_custom_model: bool) -> Result<()> {
    let config = WireConfig::parse_value(config).map_err(node_error)?;
    config
        .validate(has_custom_model, &node_capabilities())
        .map_err(node_error)
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
    let config = WireConfig::parse_value(config).map_err(node_error)?;
    config
        .validate(model.is_some(), &node_capabilities())
        .map_err(node_error)?;
    let events = EventSlot::default();
    events.set(event_handler)?;
    let model = model.map(build_model_callback).transpose()?;
    build_agent(config, model, events).map_err(node_error)
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

fn build_agent(
    config: WireConfig,
    model: Option<Arc<ModelFn>>,
    events: EventSlot,
) -> CoreResult<Agent> {
    let model = match model {
        Some(callback) => Arc::new(JsModel {
            name: if config.openai.model.is_empty() {
                "js-model".to_owned()
            } else {
                config.openai.model.clone()
            },
            context_window: config.context_window.or(config.openai.context_window),
            callback,
        }) as Arc<dyn Model>,
        None => Arc::new(build_openai(&config.openai)?) as Arc<dyn Model>,
    };

    let mut builder = CoreAgent::builder(model.clone())
        .config(config.core_config())
        .event_sink(Arc::new(JsEventSink {
            slot: events.clone(),
        }));
    if let Some(system_prompt) = config.system_prompt.clone() {
        builder = builder.system_prompt(system_prompt);
    }
    for section in config.prompt_sections.iter().cloned() {
        builder = builder.prompt_section(section);
    }
    if let Some(session_id) = config.session_id.clone() {
        builder = builder.session_id(session_id);
    }
    if let Some(skills) = &config.skills {
        let mut registry = SkillRegistry::default();
        for path in &skills.paths {
            let loaded = SkillRegistry::load_dir_with_depth(path, skills.max_depth)?;
            for skill in loaded.skills() {
                registry.insert((*skill).clone())?;
            }
        }
        builder = registry.install(builder);
    }

    let mut todos = None;
    if config.todo {
        let list = TodoList::new();
        builder = list.install(builder);
        todos = Some(list);
    }

    if config.memory.enabled {
        builder = builder
            .memory(Arc::new(InMemoryMemory::new()))
            .memory_options(config.memory_options());
        if config.memory.tools {
            builder = builder.with_memory_tools();
        }
    }

    if let Some(compression) = &config.compression {
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
    if let Some(subagent) = &config.subagent {
        let tool =
            SubAgentTool::from_agent(subagent.name.clone(), subagent.description.clone(), &inner)
                .with_max_depth(subagent.max_depth);
        inner.register_tool(Arc::new(tool))?;
    }
    Ok(Agent {
        inner: Some(inner),
        events,
        todos,
    })
}

fn build_openai(settings: &OpenAiSettings) -> CoreResult<OpenAiCompatible> {
    let mut builder = OpenAiCompatible::builder(&settings.model);
    if let Some(api_key) = &settings.api_key {
        builder = builder.api_key(api_key);
    } else if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        builder = builder.api_key(api_key);
    }
    if let Some(base_url) = &settings.base_url {
        builder = builder.base_url(base_url);
    } else if let Ok(base_url) = std::env::var("OPENAI_BASE_URL") {
        builder = builder.base_url(base_url);
    }
    if let Some(tokens) = settings.context_window {
        builder = builder.context_window(tokens);
    }
    if let Some(seconds) = settings.timeout_seconds {
        builder = builder.timeout(Duration::from_secs(seconds));
    }
    if let Some(stream_usage) = settings.stream_usage {
        builder = builder.stream_usage(stream_usage);
    }
    builder.build()
}
