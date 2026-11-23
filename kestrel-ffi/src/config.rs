use std::{collections::BTreeMap, sync::Arc, time::Duration};

use kestrel::mcp::{McpClient, McpServerConfig};
use kestrel::skills::SkillRegistry;
use kestrel::{
    Agent, AgentConfig, EventSink, InMemoryMemory, Memory, MemoryOptions, Model, OpenAiCompatible,
    SessionStore, SlidingWindowCompressor, SubAgentTool, SummarizingCompressor, TodoList,
    ToolChoice,
};
use serde::Deserialize;

use crate::{
    callbacks::{CallbackTracker, ModelCallback, StreamingModelCallback},
    model::CallbackModel,
    runtime,
};

pub(crate) struct BuiltAgent {
    pub(crate) agent: Agent,
    pub(crate) todos: Option<TodoList>,
    pub(crate) mcp_clients: Vec<McpClient>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub(crate) struct Settings {
    pub(crate) system_prompt: Option<String>,
    pub(crate) prompt_sections: Vec<String>,
    pub(crate) session_id: Option<String>,
    pub(crate) max_turns: usize,
    pub(crate) temperature: Option<f32>,
    pub(crate) max_output_tokens: Option<u32>,
    pub(crate) context_window: Option<usize>,
    pub(crate) reserve_output_tokens: usize,
    pub(crate) streaming: bool,
    pub(crate) fail_on_tool_error: bool,
    pub(crate) tool_choice: Option<ToolChoice>,
    pub(crate) todo: bool,
    pub(crate) memory: MemorySettings,
    pub(crate) compression: Option<CompressionSettings>,
    pub(crate) subagent: Option<SubagentSettings>,
    pub(crate) skills: Option<SkillsSettings>,
    pub(crate) mcp_servers: Vec<McpServerSettings>,
    pub(crate) openai: OpenAiSettings,
}

impl Default for Settings {
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
            mcp_servers: Vec::new(),
            openai: OpenAiSettings::default(),
        }
    }
}

impl Settings {
    pub(crate) fn parse(config_json: &str) -> Result<Self, String> {
        serde_json::from_str(config_json)
            .map_err(|error| format!("invalid Kestrel config JSON: {error}"))
    }

    pub(crate) fn openai_model(&self) -> Result<Arc<dyn Model>, String> {
        if self.openai.model.trim().is_empty() {
            return Err("openai.model must not be empty".to_owned());
        }
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
            if seconds == 0 {
                return Err("openai.timeout_seconds must be greater than zero".to_owned());
            }
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

    pub(crate) fn callback_model(
        &self,
        callback: ModelCallback,
        tracker: Arc<CallbackTracker>,
    ) -> Arc<dyn Model> {
        Arc::new(CallbackModel::new(
            self.callback_model_name(),
            self.context_window.or(self.openai.context_window),
            callback,
            tracker,
        ))
    }

    pub(crate) fn streaming_callback_model(
        &self,
        callback: StreamingModelCallback,
        tracker: Arc<CallbackTracker>,
    ) -> Arc<dyn Model> {
        Arc::new(CallbackModel::new_streaming(
            self.callback_model_name(),
            self.context_window.or(self.openai.context_window),
            callback,
            tracker,
        ))
    }

    fn callback_model_name(&self) -> String {
        if self.openai.model.trim().is_empty() {
            "callback".to_owned()
        } else {
            self.openai.model.clone()
        }
    }

    pub(crate) fn build(
        self,
        model: Arc<dyn Model>,
        event_sink: Arc<dyn EventSink>,
        session_store: Option<Arc<dyn SessionStore>>,
        memory_override: Option<Arc<dyn Memory>>,
    ) -> Result<BuiltAgent, String> {
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

        let mut builder = Agent::builder(model.clone())
            .config(config)
            .event_sink(event_sink);
        if let Some(session_store) = session_store {
            builder = builder.session_store(session_store);
        }
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
            let registry = load_skills(&skills)?;
            builder = registry.install(builder);
        }

        let mut todos = None;
        if self.todo {
            let list = TodoList::new();
            builder = list.install(builder);
            todos = Some(list);
        }

        if let Some(memory) = memory_override {
            builder = builder.memory(memory).memory_options(MemoryOptions {
                auto_recall: self.memory.auto_recall,
                recall_limit: self.memory.recall_limit,
                max_context_tokens: self.memory.max_context_tokens,
            });
            if self.memory.tools {
                builder = builder.with_memory_tools();
            }
        } else if self.memory.enabled {
            let memory = Arc::new(InMemoryMemory::new());
            builder = builder.memory(memory).memory_options(MemoryOptions {
                auto_recall: self.memory.auto_recall,
                recall_limit: self.memory.recall_limit,
                max_context_tokens: self.memory.max_context_tokens,
            });
            if self.memory.tools {
                builder = builder.with_memory_tools();
            }
        } else if self.memory.tools {
            return Err("memory.tools requires memory.enabled".to_owned());
        }

        if let Some(compression) = self.compression {
            if compression.max_tokens == 0 {
                return Err("compression.max_tokens must be greater than zero".to_owned());
            }
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
        if let Some(subagent) = self.subagent {
            let tool = SubAgentTool::from_agent(subagent.name, subagent.description, &agent)
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

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub(crate) struct OpenAiSettings {
    pub(crate) model: String,
    pub(crate) api_key: Option<String>,
    pub(crate) base_url: Option<String>,
    pub(crate) context_window: Option<usize>,
    pub(crate) timeout_seconds: Option<u64>,
    pub(crate) stream_usage: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub(crate) struct MemorySettings {
    pub(crate) enabled: bool,
    pub(crate) tools: bool,
    pub(crate) auto_recall: bool,
    pub(crate) recall_limit: usize,
    pub(crate) max_context_tokens: usize,
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

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct CompressionSettings {
    #[serde(default)]
    pub(crate) kind: CompressionKind,
    pub(crate) max_tokens: usize,
    pub(crate) summary_tokens: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CompressionKind {
    #[default]
    Sliding,
    Summarizing,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub(crate) struct SubagentSettings {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) max_depth: usize,
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

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub(crate) struct SkillsSettings {
    pub(crate) paths: Vec<String>,
    pub(crate) max_depth: usize,
}

impl Default for SkillsSettings {
    fn default() -> Self {
        Self {
            paths: Vec::new(),
            max_depth: 4,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub(crate) struct McpServerSettings {
    pub(crate) command: String,
    pub(crate) args: Vec<String>,
    pub(crate) env: BTreeMap<String, String>,
    pub(crate) cwd: Option<String>,
    pub(crate) request_timeout_seconds: Option<u64>,
    pub(crate) protocol_version: Option<String>,
    pub(crate) client_name: Option<String>,
    pub(crate) client_version: Option<String>,
}

fn load_skills(settings: &SkillsSettings) -> std::result::Result<SkillRegistry, String> {
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

type ConnectedMcp = (Vec<McpClient>, Vec<Arc<dyn kestrel::Tool>>);

fn connect_mcp_servers(servers: &[McpServerSettings]) -> std::result::Result<ConnectedMcp, String> {
    if servers.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let servers = servers.to_vec();
    runtime::block_on(async move {
        let mut clients = Vec::new();
        let mut tools = Vec::new();
        for server in servers {
            if server.command.trim().is_empty() {
                return Err("MCP server command must not be empty".to_owned());
            }
            let mut config = McpServerConfig::new(&server.command).args(server.args.clone());
            for (key, value) in &server.env {
                config = config.env(key, value);
            }
            if let Some(cwd) = server.cwd {
                config = config.cwd(cwd);
            }
            if let Some(seconds) = server.request_timeout_seconds {
                if seconds == 0 {
                    return Err("MCP request_timeout_seconds must be greater than zero".to_owned());
                }
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
            clients.push(client);
        }
        Ok((clients, tools))
    })?
}
