use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        Arc, Mutex as StdMutex, OnceLock, RwLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use kestrel::mcp::{McpClient, McpHttpClient, McpHttpConfig, McpServerConfig};
use kestrel::skills::SkillRegistry;
use kestrel::{
    Agent as CoreAgent, AgentEvent, ContextCompressor, EventSink, InMemoryMemory, Memory,
    MemoryEntry, MemoryQuery, Model, ModelDelta, ModelRequest, ModelResponse, OpenAiCompatible,
    Result as CoreResult, RunOptions, Session, SessionStore, SlidingWindowCompressor, SubAgentTool,
    SummarizingCompressor, TodoList, Tool, ToolContext, ToolOutput, ToolSpec,
};
use kestrel_contract::{
    AgentConfig as WireConfig, Capability, CapabilitySet, CompressionKind, McpServerSettings,
    OpenAiSettings, SCHEMA_VERSION,
};
use napi::{
    Error, Result, Status,
    bindgen_prelude::{FnArgs, Function, Promise},
    threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode, UnknownReturnValue},
};
use napi_derive::napi;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

type ModelFn = ThreadsafeFunction<
    FnArgs<(Value, String)>,
    Promise<Value>,
    FnArgs<(Value, String)>,
    Status,
    false,
>;
type ToolFn = ThreadsafeFunction<
    FnArgs<(String, Value)>,
    Promise<Value>,
    FnArgs<(String, Value)>,
    Status,
    false,
>;
type EventFn = ThreadsafeFunction<Value, UnknownReturnValue, Value, Status, false>;
type SessionLoadFn = ThreadsafeFunction<String, Promise<Value>, String, Status, false>;
type SessionSaveFn = ThreadsafeFunction<String, Promise<Value>, String, Status, false>;
type SessionDeleteFn = ThreadsafeFunction<String, Promise<Value>, String, Status, false>;
type MemoryRecallFn = ThreadsafeFunction<String, Promise<Value>, String, Status, false>;
type MemoryRememberFn = ThreadsafeFunction<String, Promise<Value>, String, Status, false>;

type ModelFunction = Function<'static, FnArgs<(Value, String)>, Promise<Value>>;
type EventFunction = Function<'static, Value, UnknownReturnValue>;
type StringPromiseFunction = Function<'static, String, Promise<Value>>;

static DELTA_SENDERS: OnceLock<
    StdMutex<HashMap<String, tokio::sync::mpsc::UnboundedSender<ModelDelta>>>,
> = OnceLock::new();
static NEXT_DELTA_ID: AtomicU64 = AtomicU64::new(1);

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
        Capability::McpStdio,
        Capability::McpHttp,
        Capability::HostTools,
        Capability::SessionStore,
        Capability::MemoryStore,
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

/// Emits one ModelDelta for an active JavaScript model invocation.
#[napi]
pub fn emit_model_delta(token: String, delta: Value) -> Result<()> {
    let delta: ModelDelta = serde_json::from_value(delta).map_err(node_error)?;
    let senders = DELTA_SENDERS.get_or_init(|| StdMutex::new(HashMap::new()));
    let sender = senders
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&token)
        .cloned();
    if let Some(sender) = sender {
        let _ = sender.send(delta);
    }
    Ok(())
}

fn build_model_callback(
    callback: Function<'static, FnArgs<(Value, String)>, Promise<Value>>,
) -> Result<Arc<ModelFn>> {
    let callback: ModelFn = callback
        .build_threadsafe_function::<FnArgs<(Value, String)>>()
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

fn build_session_load_callback(
    callback: Function<'static, String, Promise<Value>>,
) -> Result<Arc<SessionLoadFn>> {
    let callback: SessionLoadFn = callback
        .build_threadsafe_function::<String>()
        .callee_handled::<false>()
        .build()?;
    Ok(Arc::new(callback))
}

fn build_session_save_callback(
    callback: Function<'static, String, Promise<Value>>,
) -> Result<Arc<SessionSaveFn>> {
    let callback: SessionSaveFn = callback
        .build_threadsafe_function::<String>()
        .callee_handled::<false>()
        .build()?;
    Ok(Arc::new(callback))
}

fn build_session_delete_callback(
    callback: Function<'static, String, Promise<Value>>,
) -> Result<Arc<SessionDeleteFn>> {
    let callback: SessionDeleteFn = callback
        .build_threadsafe_function::<String>()
        .callee_handled::<false>()
        .build()?;
    Ok(Arc::new(callback))
}

fn build_memory_recall_callback(
    callback: Function<'static, String, Promise<Value>>,
) -> Result<Arc<MemoryRecallFn>> {
    let callback: MemoryRecallFn = callback
        .build_threadsafe_function::<String>()
        .callee_handled::<false>()
        .build()?;
    Ok(Arc::new(callback))
}

fn build_memory_remember_callback(
    callback: Function<'static, String, Promise<Value>>,
) -> Result<Arc<MemoryRememberFn>> {
    let callback: MemoryRememberFn = callback
        .build_threadsafe_function::<String>()
        .callee_handled::<false>()
        .build()?;
    Ok(Arc::new(callback))
}

struct JsSessionStore {
    load: Arc<SessionLoadFn>,
    save: Arc<SessionSaveFn>,
    delete: Option<Arc<SessionDeleteFn>>,
}

#[async_trait]
impl SessionStore for JsSessionStore {
    async fn load(&self, id: &str) -> CoreResult<Option<Session>> {
        let promise = self
            .load
            .call_async(id.to_owned())
            .await
            .map_err(|error| kestrel::Error::Protocol(error.to_string()))?;
        let value = promise
            .await
            .map_err(|error| kestrel::Error::Protocol(error.to_string()))?;
        if value.is_null() {
            Ok(None)
        } else {
            let json = value.as_str().ok_or_else(|| {
                kestrel::Error::Protocol(
                    "JavaScript session loader must return JSON or null".to_owned(),
                )
            })?;
            serde_json::from_str(json).map(Some).map_err(|error| {
                kestrel::Error::Protocol(format!("invalid JavaScript session: {error}"))
            })
        }
    }

    async fn save(&self, session: &Session) -> CoreResult<()> {
        let json = serde_json::to_string(session)?;
        let promise = self
            .save
            .call_async(json)
            .await
            .map_err(|error| kestrel::Error::Protocol(error.to_string()))?;
        promise
            .await
            .map_err(|error| kestrel::Error::Protocol(error.to_string()))?;
        Ok(())
    }

    async fn delete(&self, id: &str) -> CoreResult<()> {
        let Some(delete) = &self.delete else {
            return Ok(());
        };
        let promise = delete
            .call_async(id.to_owned())
            .await
            .map_err(|error| kestrel::Error::Protocol(error.to_string()))?;
        promise
            .await
            .map_err(|error| kestrel::Error::Protocol(error.to_string()))?;
        Ok(())
    }
}

struct JsMemory {
    recall: Arc<MemoryRecallFn>,
    remember: Arc<MemoryRememberFn>,
}

#[async_trait]
impl Memory for JsMemory {
    async fn remember(&self, entry: MemoryEntry) -> CoreResult<()> {
        let json = serde_json::to_string(&entry)?;
        let promise = self
            .remember
            .call_async(json)
            .await
            .map_err(|error| kestrel::Error::Memory(error.to_string()))?;
        promise
            .await
            .map_err(|error| kestrel::Error::Memory(error.to_string()))?;
        Ok(())
    }

    async fn recall(&self, query: &MemoryQuery) -> CoreResult<Vec<MemoryEntry>> {
        let json = serde_json::to_string(query)?;
        let promise = self
            .recall
            .call_async(json)
            .await
            .map_err(|error| kestrel::Error::Memory(error.to_string()))?;
        let value = promise
            .await
            .map_err(|error| kestrel::Error::Memory(error.to_string()))?;
        let json = value.as_str().ok_or_else(|| {
            kestrel::Error::Memory("JavaScript memory recall must return a JSON string".to_owned())
        })?;
        serde_json::from_str(json).map_err(|error| {
            kestrel::Error::Memory(format!("invalid JavaScript memory entries: {error}"))
        })
    }
}

struct JsModel {
    name: String,
    context_window: Option<usize>,
    callback: Arc<ModelFn>,
}

struct DeltaRegistration {
    token: String,
}

impl Drop for DeltaRegistration {
    fn drop(&mut self) {
        if let Some(senders) = DELTA_SENDERS.get() {
            senders
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .remove(&self.token);
        }
    }
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
        self.complete_internal(request, None).await
    }

    async fn complete_stream(
        &self,
        request: ModelRequest,
        on_delta: &(dyn Fn(ModelDelta) + Send + Sync),
    ) -> CoreResult<ModelResponse> {
        self.complete_internal(request, Some(on_delta)).await
    }
}

impl JsModel {
    async fn complete_internal(
        &self,
        request: ModelRequest,
        on_delta: Option<&(dyn Fn(ModelDelta) + Send + Sync)>,
    ) -> CoreResult<ModelResponse> {
        let request_value = serde_json::to_value(request)
            .map_err(|error| kestrel::Error::Model(error.to_string()))?;
        let token = NEXT_DELTA_ID.fetch_add(1, Ordering::Relaxed).to_string();
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        DELTA_SENDERS
            .get_or_init(|| StdMutex::new(HashMap::new()))
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(token.clone(), sender);
        let _registration = DeltaRegistration {
            token: token.clone(),
        };
        let promise = self
            .callback
            .call_async(FnArgs::from((request_value, token)))
            .await
            .map_err(|error| kestrel::Error::Model(error.to_string()))?;
        tokio::pin!(promise);
        let mut channel_open = true;
        let response_value = loop {
            tokio::select! {
                delta = receiver.recv(), if channel_open => {
                    match delta {
                        Some(delta) => {
                            if let Some(on_delta) = on_delta {
                                on_delta(delta);
                            }
                        }
                        None => channel_open = false,
                    }
                }
                result = &mut promise => {
                    break result.map_err(|error| kestrel::Error::Model(error.to_string()))?;
                }
            }
        };
        while let Ok(delta) = receiver.try_recv() {
            if let Some(on_delta) = on_delta {
                on_delta(delta);
            }
        }
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
    mcp_clients: StdMutex<Vec<NodeMcpClient>>,
    mcp_tools: StdMutex<Vec<String>>,
}

enum NodeMcpClient {
    Stdio(McpClient),
    Http(McpHttpClient),
}

impl NodeMcpClient {
    async fn close(&self) {
        match self {
            Self::Stdio(client) => {
                let _ = client.close().await;
            }
            Self::Http(client) => {
                let _ = client.close().await;
            }
        }
    }
}

struct RunState {
    cancellation: CancellationToken,
    task: StdMutex<Option<tokio::task::JoinHandle<()>>>,
    result: StdMutex<Option<tokio::sync::oneshot::Receiver<std::result::Result<Value, String>>>>,
}

/// A cancellable handle for one asynchronous run.
#[napi]
pub struct Run {
    state: Arc<RunState>,
}

#[napi]
impl Run {
    /// Requests cancellation of this run.
    #[napi]
    pub fn cancel(&self) {
        self.state.cancellation.cancel();
    }

    /// Returns whether the native task has finished.
    #[napi]
    pub fn is_done(&self) -> bool {
        self.state
            .task
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .as_ref()
            .is_none_or(tokio::task::JoinHandle::is_finished)
    }

    /// Waits for the final output. Can be called once.
    #[napi]
    pub async fn result(&self) -> Result<Value> {
        let receiver = self
            .state
            .result
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
            .ok_or_else(|| node_error("run result has already been consumed"))?;
        match receiver.await {
            Ok(Ok(output)) => Ok(output),
            Ok(Err(error)) => Err(node_error(error)),
            Err(error) => Err(node_error(format!(
                "run task ended without output: {error}"
            ))),
        }
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        self.state.cancellation.cancel();
    }
}

/// Creates a Kestrel agent.
#[napi]
#[allow(clippy::too_many_arguments)]
pub fn create_agent(
    config: Value,
    model: Option<ModelFunction>,
    event_handler: Option<EventFunction>,
    session_load: Option<StringPromiseFunction>,
    session_save: Option<StringPromiseFunction>,
    session_delete: Option<StringPromiseFunction>,
    memory_recall: Option<StringPromiseFunction>,
    memory_remember: Option<StringPromiseFunction>,
) -> Result<Agent> {
    let inputs = prepare_agent_inputs(
        config,
        model,
        event_handler,
        session_load,
        session_save,
        session_delete,
        memory_recall,
        memory_remember,
    )?;
    if !inputs.config.mcp_servers.is_empty() {
        return Err(node_error("MCP configuration requires createAgentAsync()"));
    }
    build_agent(
        inputs.config,
        inputs.model,
        inputs.events,
        inputs.session_store,
        inputs.memory,
    )
    .map_err(node_error)
}

struct AgentInputs {
    config: WireConfig,
    model: Option<Arc<ModelFn>>,
    events: EventSlot,
    session_store: Option<Arc<dyn SessionStore>>,
    memory: Option<Arc<dyn Memory>>,
}

#[allow(clippy::too_many_arguments)]
fn prepare_agent_inputs(
    config: Value,
    model: Option<ModelFunction>,
    event_handler: Option<EventFunction>,
    session_load: Option<StringPromiseFunction>,
    session_save: Option<StringPromiseFunction>,
    session_delete: Option<StringPromiseFunction>,
    memory_recall: Option<StringPromiseFunction>,
    memory_remember: Option<StringPromiseFunction>,
) -> Result<AgentInputs> {
    let mut config = WireConfig::parse_value(config).map_err(node_error)?;
    if memory_recall.is_some() || memory_remember.is_some() {
        config.memory.enabled = true;
    }
    config
        .validate(model.is_some(), &node_capabilities())
        .map_err(node_error)?;
    let events = EventSlot::default();
    events.set(event_handler)?;
    let model = model.map(build_model_callback).transpose()?;
    if session_load.is_some() != session_save.is_some() {
        return Err(node_error(
            "sessionStore must provide both load and save methods",
        ));
    }
    let session_store = match (session_load, session_save) {
        (Some(load), Some(save)) => Some(Arc::new(JsSessionStore {
            load: build_session_load_callback(load)?,
            save: build_session_save_callback(save)?,
            delete: session_delete
                .map(build_session_delete_callback)
                .transpose()?,
        }) as Arc<dyn SessionStore>),
        _ => None,
    };
    let memory = match (memory_recall, memory_remember) {
        (Some(recall), Some(remember)) => Some(Arc::new(JsMemory {
            recall: build_memory_recall_callback(recall)?,
            remember: build_memory_remember_callback(remember)?,
        }) as Arc<dyn Memory>),
        (None, None) => None,
        _ => {
            return Err(node_error(
                "memory must provide both recall and remember methods",
            ));
        }
    };
    Ok(AgentInputs {
        config,
        model,
        events,
        session_store,
        memory,
    })
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

    /// Starts a run and immediately returns a cancellable handle.
    #[napi]
    pub fn start_run(&self, prompt: String) -> Result<Run> {
        let inner = self.core()?.clone();
        let cancellation = CancellationToken::new();
        let run_token = cancellation.clone();
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let task = napi::bindgen_prelude::spawn(async move {
            let output = inner
                .run_with(prompt, RunOptions::default().with_cancellation(run_token))
                .await
                .map_err(|error| error.to_string())
                .and_then(|output| serde_json::to_value(output).map_err(|error| error.to_string()));
            let _ = sender.send(output);
        });
        Ok(Run {
            state: Arc::new(RunState {
                cancellation,
                task: StdMutex::new(Some(task)),
                result: StdMutex::new(Some(receiver)),
            }),
        })
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

    /// Deletes the stored session and clears local history.
    #[napi]
    pub async fn delete_session(&self) -> Result<()> {
        self.core()?
            .clone()
            .delete_session()
            .await
            .map_err(node_error)
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

    /// Connects MCP servers and registers their tools.
    #[napi]
    pub async fn connect_mcp(&self, servers: Value) -> Result<()> {
        let servers: Vec<McpServerSettings> =
            serde_json::from_value(servers).map_err(node_error)?;
        let core = self.core()?.clone();
        let (clients, tools) = connect_mcp_servers(&servers).await.map_err(node_error)?;
        let names = tools
            .iter()
            .map(|tool| tool.spec().name)
            .collect::<Vec<_>>();
        if let Err(error) = core.tools().register_all(tools) {
            for client in &clients {
                client.close().await;
            }
            return Err(node_error(error));
        }
        self.mcp_tools
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .extend(names);
        self.mcp_clients
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .extend(clients);
        Ok(())
    }

    /// Closes MCP connections while keeping the agent usable.
    #[napi]
    pub async fn close_mcp(&self) -> Result<()> {
        let names = std::mem::take(
            &mut *self
                .mcp_tools
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
        );
        if let Some(core) = &self.inner {
            for name in names {
                core.tools().remove(&name);
            }
        }
        let clients = std::mem::take(
            &mut *self
                .mcp_clients
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
        );
        for client in clients {
            client.close().await;
        }
        Ok(())
    }

    /// Releases native resources. All runs must be awaited first.
    #[napi]
    pub fn close(&mut self) {
        self.release();
        let clients = std::mem::take(
            &mut *self
                .mcp_clients
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
        );
        if !clients.is_empty() {
            napi::bindgen_prelude::spawn(async move {
                for client in clients {
                    client.close().await;
                }
            });
        }
    }

    fn release(&mut self) {
        self.inner = None;
        self.todos = None;
        self.events.set(None).ok();
    }
}

fn build_agent(
    config: WireConfig,
    model: Option<Arc<ModelFn>>,
    events: EventSlot,
    session_store: Option<Arc<dyn SessionStore>>,
    memory: Option<Arc<dyn Memory>>,
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
    if let Some(session_store) = session_store {
        builder = builder.session_store(session_store);
    }
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

    if let Some(memory) = memory {
        builder = builder
            .memory(memory)
            .memory_options(config.memory_options());
        if config.memory.tools {
            builder = builder.with_memory_tools();
        }
    } else if config.memory.enabled {
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
        mcp_clients: StdMutex::new(Vec::new()),
        mcp_tools: StdMutex::new(Vec::new()),
    })
}

async fn connect_mcp_servers(
    servers: &[McpServerSettings],
) -> CoreResult<(Vec<NodeMcpClient>, Vec<Arc<dyn Tool>>)> {
    let mut clients = Vec::new();
    let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
    for server in servers {
        match connect_mcp_server(server).await {
            Ok((client, discovered)) => {
                clients.push(client);
                tools.extend(discovered);
            }
            Err(error) => {
                for client in &clients {
                    client.close().await;
                }
                return Err(error);
            }
        }
    }
    Ok((clients, tools))
}

async fn connect_mcp_server(
    server: &McpServerSettings,
) -> CoreResult<(NodeMcpClient, Vec<Arc<dyn Tool>>)> {
    if let Some(url) = &server.url {
        let mut config = McpHttpConfig::new(url.clone());
        config.headers = server.headers.clone();
        if let Some(seconds) = server.request_timeout_seconds {
            config = config.request_timeout(Duration::from_secs(seconds));
        }
        if let Some(version) = &server.protocol_version {
            config = config.protocol_version(version.clone());
        }
        if let Some(name) = &server.client_name {
            config.client_name = name.clone();
        }
        if let Some(version) = &server.client_version {
            config.client_version = version.clone();
        }
        let client = McpHttpClient::connect(config).await?;
        let tools = client.tools().await?;
        Ok((NodeMcpClient::Http(client), tools))
    } else {
        let mut config = McpServerConfig::new(server.command.clone()).args(server.args.clone());
        for (key, value) in &server.env {
            config = config.env(key, value);
        }
        if let Some(cwd) = &server.cwd {
            config = config.cwd(PathBuf::from(cwd));
        }
        if let Some(seconds) = server.request_timeout_seconds {
            config = config.request_timeout(Duration::from_secs(seconds));
        }
        if let Some(version) = &server.protocol_version {
            config = config.protocol_version(version.clone());
        }
        if let Some(name) = &server.client_name {
            config.client_name = name.clone();
        }
        if let Some(version) = &server.client_version {
            config.client_version = version.clone();
        }
        let client = McpClient::connect(config).await?;
        let tools = client.tools().await?;
        Ok((NodeMcpClient::Stdio(client), tools))
    }
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
