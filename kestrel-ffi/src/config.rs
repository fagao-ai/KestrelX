use std::{sync::Arc, time::Duration};

use kestrel::mcp::{McpClient, McpHttpClient, McpHttpConfig, McpServerConfig};
use kestrel::skills::SkillRegistry;
use kestrel::{
    Agent, EventSink, InMemoryMemory, Memory, Model, OpenAiCompatible, SessionStore,
    SlidingWindowCompressor, SubAgentTool, SummarizingCompressor, TodoList,
};
pub(crate) use kestrel_contract::AgentConfig as Settings;
use kestrel_contract::{
    Capability, CapabilitySet, CompressionKind, McpServerSettings, SkillsSettings,
};

use crate::{
    callbacks::{CallbackTracker, ModelCallback, StreamingModelCallback},
    model::CallbackModel,
    runtime,
};

pub(crate) struct BuiltAgent {
    pub(crate) agent: Agent,
    pub(crate) todos: Option<TodoList>,
    pub(crate) mcp_clients: Vec<McpConnection>,
}

#[derive(Clone)]
pub(crate) enum McpConnection {
    Stdio(McpClient),
    Http(McpHttpClient),
}

impl McpConnection {
    pub(crate) async fn close(&self) {
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

pub(crate) fn capabilities() -> CapabilitySet {
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

pub(crate) fn parse_settings(
    config_json: &str,
    has_custom_model: bool,
) -> Result<Settings, String> {
    Settings::parse_and_validate_json(config_json, has_custom_model, &capabilities())
        .map_err(|error| error.to_string())
}

pub(crate) trait SettingsExt {
    fn openai_model(&self) -> Result<Arc<dyn Model>, String>;
    fn callback_model(
        &self,
        callback: ModelCallback,
        tracker: Arc<CallbackTracker>,
    ) -> Arc<dyn Model>;
    fn streaming_callback_model(
        &self,
        callback: StreamingModelCallback,
        tracker: Arc<CallbackTracker>,
    ) -> Arc<dyn Model>;
    fn build(
        self,
        model: Arc<dyn Model>,
        event_sink: Arc<dyn EventSink>,
        session_store: Option<Arc<dyn SessionStore>>,
        memory_override: Option<Arc<dyn Memory>>,
    ) -> Result<BuiltAgent, String>;
}

impl SettingsExt for Settings {
    fn openai_model(&self) -> Result<Arc<dyn Model>, String> {
        let mut builder = OpenAiCompatible::builder(&self.openai.model);
        if let Some(api_key) = &self.openai.api_key {
            builder = builder.api_key(api_key);
        } else if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            builder = builder.api_key(api_key);
        }
        if let Some(base_url) = &self.openai.base_url {
            builder = builder.base_url(base_url);
        } else if let Ok(base_url) = std::env::var("OPENAI_BASE_URL") {
            builder = builder.base_url(base_url);
        }
        if let Some(tokens) = self.openai.context_window {
            builder = builder.context_window(tokens);
        }
        if let Some(seconds) = self.openai.timeout_seconds {
            builder = builder.timeout(Duration::from_secs(seconds));
        }
        if let Some(include_usage) = self.openai.stream_usage {
            builder = builder.stream_usage(include_usage);
        }
        let model = builder
            .build()
            .map_err(|error| format!("failed to build OpenAI provider: {error}"))?;
        Ok(Arc::new(model))
    }

    fn callback_model(
        &self,
        callback: ModelCallback,
        tracker: Arc<CallbackTracker>,
    ) -> Arc<dyn Model> {
        Arc::new(CallbackModel::new(
            callback_model_name(self),
            self.context_window.or(self.openai.context_window),
            callback,
            tracker,
        ))
    }

    fn streaming_callback_model(
        &self,
        callback: StreamingModelCallback,
        tracker: Arc<CallbackTracker>,
    ) -> Arc<dyn Model> {
        Arc::new(CallbackModel::new_streaming(
            callback_model_name(self),
            self.context_window.or(self.openai.context_window),
            callback,
            tracker,
        ))
    }

    fn build(
        self,
        model: Arc<dyn Model>,
        event_sink: Arc<dyn EventSink>,
        session_store: Option<Arc<dyn SessionStore>>,
        memory_override: Option<Arc<dyn Memory>>,
    ) -> Result<BuiltAgent, String> {
        let mut builder = Agent::builder(model.clone())
            .config(self.core_config())
            .event_sink(event_sink);
        if let Some(session_store) = session_store {
            builder = builder.session_store(session_store);
        }
        if let Some(system_prompt) = self.system_prompt.clone() {
            builder = builder.system_prompt(system_prompt);
        }
        for section in self.prompt_sections.iter().cloned() {
            builder = builder.prompt_section(section);
        }
        if let Some(session_id) = self.session_id.clone() {
            builder = builder.session_id(session_id);
        }
        if let Some(skills) = &self.skills {
            builder = load_skills(skills)?.install(builder);
        }

        let mut todos = None;
        if self.todo {
            let list = TodoList::new();
            builder = list.install(builder);
            todos = Some(list);
        }

        let memory_options = self.memory_options();
        if let Some(memory) = memory_override {
            if !self.memory.enabled {
                return Err(
                    "memory.enabled must be true when host memory callbacks are installed"
                        .to_owned(),
                );
            }
            builder = builder.memory(memory).memory_options(memory_options);
            if self.memory.tools {
                builder = builder.with_memory_tools();
            }
        } else if self.memory.enabled {
            builder = builder
                .memory(Arc::new(InMemoryMemory::new()))
                .memory_options(memory_options);
            if self.memory.tools {
                builder = builder.with_memory_tools();
            }
        }

        if let Some(compression) = &self.compression {
            builder = match compression.kind {
                CompressionKind::Sliding => builder.compressor(Arc::new(
                    SlidingWindowCompressor::new(compression.max_tokens),
                )),
                CompressionKind::Summarizing => {
                    let mut compressor = SummarizingCompressor::new(model, compression.max_tokens);
                    if let Some(tokens) = compression.summary_tokens {
                        compressor = compressor.with_summary_tokens(tokens);
                    }
                    builder.compressor(Arc::new(compressor))
                }
            };
        }

        let agent = builder
            .build()
            .map_err(|error| format!("failed to build agent: {error}"))?;
        if let Some(subagent) = &self.subagent {
            let tool = SubAgentTool::from_agent(
                subagent.name.clone(),
                subagent.description.clone(),
                &agent,
            )
            .with_max_depth(subagent.max_depth);
            agent
                .register_tool(Arc::new(tool))
                .map_err(|error| format!("failed to register sub-agent: {error}"))?;
        }

        let (mcp_clients, mcp_tools) = connect_mcp_servers(&self.mcp_servers)?;
        agent
            .tools()
            .register_all(mcp_tools)
            .map_err(|error| format!("failed to register MCP tools: {error}"))?;

        Ok(BuiltAgent {
            agent,
            todos,
            mcp_clients,
        })
    }
}

fn callback_model_name(settings: &Settings) -> String {
    if settings.openai.model.trim().is_empty() {
        "callback".to_owned()
    } else {
        settings.openai.model.clone()
    }
}

fn load_skills(settings: &SkillsSettings) -> Result<SkillRegistry, String> {
    let mut combined = SkillRegistry::default();
    for path in &settings.paths {
        let registry = SkillRegistry::load_dir_with_depth(path, settings.max_depth)
            .map_err(|error| format!("failed to load skills from `{path}`: {error}"))?;
        for skill in registry.skills() {
            combined
                .insert((*skill).clone())
                .map_err(|error| format!("failed to merge skills: {error}"))?;
        }
    }
    Ok(combined)
}

type ConnectedMcp = (Vec<McpConnection>, Vec<Arc<dyn kestrel::Tool>>);

fn connect_mcp_servers(servers: &[McpServerSettings]) -> std::result::Result<ConnectedMcp, String> {
    if servers.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let servers = servers.to_vec();
    runtime::block_on(async move {
        let mut clients = Vec::new();
        let mut tools = Vec::new();
        for server in servers {
            if let Some(url) = server.url {
                let mut config = McpHttpConfig::new(url);
                config.headers = server.headers.clone();
                if let Some(seconds) = server.request_timeout_seconds {
                    config = config.request_timeout(Duration::from_secs(seconds));
                }
                if let Some(version) = server.protocol_version {
                    config = config.protocol_version(version);
                }
                if let Some(name) = server.client_name {
                    config.client_name = name;
                }
                if let Some(version) = server.client_version {
                    config.client_version = version;
                }
                let client = McpHttpClient::connect(config)
                    .await
                    .map_err(|error| format!("MCP HTTP connect failed: {error}"))?;
                let discovered = client
                    .tools()
                    .await
                    .map_err(|error| format!("MCP HTTP tools/list failed: {error}"))?;
                tools.extend(discovered);
                clients.push(McpConnection::Http(client));
            } else {
                let mut config = McpServerConfig::new(&server.command).args(server.args.clone());
                for (key, value) in &server.env {
                    config = config.env(key, value);
                }
                if let Some(cwd) = server.cwd {
                    config = config.cwd(cwd);
                }
                if let Some(seconds) = server.request_timeout_seconds {
                    config = config.request_timeout(Duration::from_secs(seconds));
                }
                if let Some(version) = server.protocol_version {
                    config = config.protocol_version(version);
                }
                if let Some(name) = server.client_name {
                    config.client_name = name;
                }
                if let Some(version) = server.client_version {
                    config.client_version = version;
                }

                let client = McpClient::connect(config)
                    .await
                    .map_err(|error| format!("MCP connect failed: {error}"))?;
                let discovered = client
                    .tools()
                    .await
                    .map_err(|error| format!("MCP tools/list failed: {error}"))?;
                tools.extend(discovered);
                clients.push(McpConnection::Stdio(client));
            }
        }
        Ok((clients, tools))
    })?
}
