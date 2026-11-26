use std::collections::BTreeMap;

use kestrel::{
    MemoryCategory, MemoryOptions, ProviderConfig, ProviderKind, ToolChoice, ToolConcurrency,
};
use serde::{Deserialize, Serialize};

use crate::{Capability, CapabilitySet, ContractError, SCHEMA_VERSION};

/// Versioned, binding-neutral agent configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgentConfig {
    /// Configuration schema version.
    pub schema_version: u32,
    /// Base system prompt.
    pub system_prompt: Option<String>,
    /// Additional prompt sections.
    pub prompt_sections: Vec<String>,
    /// Stable session identifier.
    pub session_id: Option<String>,
    /// Maximum model turns in one run.
    pub max_turns: usize,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Maximum output tokens per call.
    pub max_output_tokens: Option<u32>,
    /// Model context window.
    pub context_window: Option<usize>,
    /// Output tokens reserved inside the context window.
    pub reserve_output_tokens: usize,
    /// Request incremental model output.
    pub streaming: bool,
    /// Abort on recoverable tool errors.
    pub fail_on_tool_error: bool,
    /// Tool selection policy.
    pub tool_choice: Option<ToolChoice>,
    /// Per-tool execution timeout in milliseconds.
    pub tool_timeout_ms: Option<u64>,
    /// Maximum retained tool output characters.
    pub max_tool_output_chars: Option<usize>,
    /// Tool call concurrency policy.
    pub tool_concurrency: ToolConcurrency,
    /// Install todo tools.
    pub todo: bool,
    /// Long-term memory settings.
    pub memory: MemorySettings,
    /// Context compression settings.
    pub compression: Option<CompressionSettings>,
    /// Sub-agent tool settings.
    pub subagent: Option<SubagentSettings>,
    /// Skill discovery settings.
    pub skills: Option<SkillsSettings>,
    /// MCP server settings.
    pub mcp_servers: Vec<McpServerSettings>,
    /// OpenAI-compatible provider settings.
    pub openai: OpenAiSettings,
    /// Built-in provider selection. Overrides `openai` when present.
    pub provider: Option<ProviderConfig>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
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
            tool_timeout_ms: None,
            max_tool_output_chars: None,
            tool_concurrency: ToolConcurrency::Serial,
            todo: false,
            memory: MemorySettings::default(),
            compression: None,
            subagent: None,
            skills: None,
            mcp_servers: Vec::new(),
            openai: OpenAiSettings::default(),
            provider: None,
        }
    }
}

impl AgentConfig {
    /// Parses JSON without capability validation.
    pub fn parse_json(input: &str) -> Result<Self, ContractError> {
        Ok(serde_json::from_str(input)?)
    }

    /// Parses a JSON value without capability validation.
    pub fn parse_value(value: serde_json::Value) -> Result<Self, ContractError> {
        Ok(serde_json::from_value(value)?)
    }

    /// Parses JSON and checks it against a binding's capabilities.
    pub fn parse_and_validate_json(
        input: &str,
        has_custom_model: bool,
        supported: &CapabilitySet,
    ) -> Result<Self, ContractError> {
        let config = Self::parse_json(input)?;
        config.validate(has_custom_model, supported)?;
        Ok(config)
    }

    /// Validates structure and required binding capabilities.
    pub fn validate(
        &self,
        has_custom_model: bool,
        supported: &CapabilitySet,
    ) -> Result<(), ContractError> {
        self.validate_values(has_custom_model)?;
        let required = self.required_capabilities(has_custom_model);
        let missing = supported.missing(&required);
        if !missing.is_empty() {
            return Err(ContractError::UnsupportedCapabilities(
                missing
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", "),
            ));
        }
        Ok(())
    }

    /// Returns capabilities required by this configuration.
    pub fn required_capabilities(&self, has_custom_model: bool) -> CapabilitySet {
        let mut capabilities = Vec::new();
        capabilities.push(if has_custom_model {
            Capability::CustomModel
        } else {
            match self
                .provider
                .as_ref()
                .map(|provider| provider.kind)
                .unwrap_or(ProviderKind::OpenAiChat)
            {
                ProviderKind::OpenAiChat => Capability::OpenAi,
                ProviderKind::OpenAiResponses => Capability::OpenAiResponses,
                ProviderKind::Anthropic => Capability::Anthropic,
                ProviderKind::Gemini => Capability::Gemini,
            }
        });
        if self.streaming {
            capabilities.push(Capability::Streaming);
        }
        if self.todo {
            capabilities.push(Capability::Todo);
        }
        if self.memory.enabled || self.memory.tools {
            capabilities.push(Capability::Memory);
        }
        if let Some(compression) = &self.compression {
            capabilities.push(match compression.kind {
                CompressionKind::Sliding => Capability::CompressionSliding,
                CompressionKind::Summarizing => Capability::CompressionSummarizing,
            });
        }
        if self.subagent.is_some() {
            capabilities.push(Capability::Subagent);
        }
        if self
            .skills
            .as_ref()
            .is_some_and(|skills| !skills.paths.is_empty())
        {
            capabilities.push(Capability::Skills);
        }
        for server in &self.mcp_servers {
            capabilities.push(if server.url.is_some() {
                Capability::McpHttp
            } else {
                Capability::McpStdio
            });
        }
        CapabilitySet::new(capabilities)
    }

    /// Converts this wire config into the core runtime config.
    pub fn core_config(&self) -> kestrel::AgentConfig {
        let mut config = kestrel::AgentConfig {
            max_turns: self.max_turns,
            temperature: self.temperature,
            max_output_tokens: self.max_output_tokens,
            context_window: self.context_window,
            reserve_output_tokens: self.reserve_output_tokens,
            streaming: self.streaming,
            fail_on_tool_error: self.fail_on_tool_error,
            tool_timeout_ms: self.tool_timeout_ms,
            max_tool_output_chars: self.max_tool_output_chars,
            tool_concurrency: self.tool_concurrency,
            ..Default::default()
        };
        if let Some(tool_choice) = &self.tool_choice {
            config.tool_choice = tool_choice.clone();
        }
        config
    }

    /// Returns core memory options.
    pub fn memory_options(&self) -> MemoryOptions {
        MemoryOptions {
            auto_recall: self.memory.auto_recall,
            recall_limit: self.memory.recall_limit,
            max_context_tokens: self.memory.max_context_tokens,
            read_categories: self.memory.read_categories.clone(),
            write_category: self.memory.write_category.clone(),
        }
    }

    /// Returns the selected provider, falling back to legacy `openai` settings.
    pub fn provider_config(&self) -> ProviderConfig {
        self.provider.clone().unwrap_or_else(|| ProviderConfig {
            kind: ProviderKind::OpenAiChat,
            model: self.openai.model.clone(),
            api_key: self.openai.api_key.clone(),
            base_url: self.openai.base_url.clone(),
            api_version: None,
            context_window: self.openai.context_window,
            timeout_seconds: self.openai.timeout_seconds,
            stream_usage: self.openai.stream_usage,
            headers: BTreeMap::new(),
        })
    }

    /// Model name used for diagnostics and callback-only configurations.
    pub fn model_name(&self) -> &str {
        self.provider
            .as_ref()
            .map(|provider| provider.model.as_str())
            .filter(|model| !model.trim().is_empty())
            .or_else(|| {
                (!self.openai.model.trim().is_empty()).then_some(self.openai.model.as_str())
            })
            .unwrap_or("callback")
    }

    /// Context window declared by the selected model configuration.
    pub fn model_context_window(&self) -> Option<usize> {
        match &self.provider {
            Some(provider) => provider.context_window,
            None => self.openai.context_window,
        }
    }

    fn validate_values(&self, has_custom_model: bool) -> Result<(), ContractError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(ContractError::UnsupportedSchema {
                found: self.schema_version,
                expected: SCHEMA_VERSION,
            });
        }
        if self.max_turns == 0 {
            return Err(invalid("max_turns must be at least 1"));
        }
        if let Some(window) = self.context_window {
            if window == 0 {
                return Err(invalid("context_window must be greater than zero"));
            }
            if self.reserve_output_tokens >= window {
                return Err(invalid(
                    "reserve_output_tokens must be smaller than context_window",
                ));
            }
        }
        if self.max_output_tokens == Some(0) {
            return Err(invalid("max_output_tokens must be greater than zero"));
        }
        if self.tool_timeout_ms == Some(0) {
            return Err(invalid("tool_timeout_ms must be greater than zero"));
        }
        if self.max_tool_output_chars == Some(0) {
            return Err(invalid("max_tool_output_chars must be greater than zero"));
        }
        if let Some(temperature) = self.temperature
            && (!temperature.is_finite() || temperature < 0.0)
        {
            return Err(invalid("temperature must be a finite non-negative number"));
        }
        if self
            .session_id
            .as_ref()
            .is_some_and(|id| id.trim().is_empty())
        {
            return Err(invalid("session_id must not be empty"));
        }
        if self.memory.tools && !self.memory.enabled {
            return Err(invalid("memory.tools requires memory.enabled"));
        }
        for category in self
            .memory
            .read_categories
            .iter()
            .chain(self.memory.write_category.iter())
        {
            if category.name.trim().is_empty() {
                return Err(invalid("memory category name must not be empty"));
            }
        }
        if let Some(compression) = &self.compression {
            if compression.max_tokens == 0 {
                return Err(invalid("compression.max_tokens must be greater than zero"));
            }
            if compression.summary_tokens == Some(0) {
                return Err(invalid(
                    "compression.summary_tokens must be greater than zero",
                ));
            }
        }
        if let Some(subagent) = &self.subagent {
            if subagent.name.trim().is_empty() {
                return Err(invalid("subagent.name must not be empty"));
            }
            if subagent.max_depth == 0 {
                return Err(invalid("subagent.max_depth must be greater than zero"));
            }
        }
        if let Some(skills) = &self.skills
            && skills.max_depth == 0
            && skills.paths.is_empty()
        {
            return Err(invalid("skills.max_depth must be greater than zero"));
        }
        for (index, server) in self.mcp_servers.iter().enumerate() {
            server.validate(index)?;
        }
        if let Some(provider) = &self.provider {
            provider
                .validate()
                .map_err(|error| invalid(&error.to_string()))?;
        }
        if !has_custom_model
            && self
                .provider
                .as_ref()
                .map(|provider| provider.model.trim().is_empty())
                .unwrap_or_else(|| self.openai.model.trim().is_empty())
        {
            return Err(invalid(
                "provider.model or openai.model is required when no custom model is supplied",
            ));
        }
        if self.provider.is_none() && self.openai.timeout_seconds == Some(0) {
            return Err(invalid("openai.timeout_seconds must be greater than zero"));
        }
        Ok(())
    }
}

fn invalid(message: &str) -> ContractError {
    ContractError::Invalid(message.to_owned())
}

/// OpenAI-compatible provider settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OpenAiSettings {
    /// Model name.
    pub model: String,
    /// API key.
    pub api_key: Option<String>,
    /// API base URL.
    pub base_url: Option<String>,
    /// Context window.
    pub context_window: Option<usize>,
    /// HTTP timeout.
    pub timeout_seconds: Option<u64>,
    /// Request usage in streamed responses.
    pub stream_usage: Option<bool>,
}

/// Long-term memory settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MemorySettings {
    /// Enable memory.
    pub enabled: bool,
    /// Register memory tools.
    pub tools: bool,
    /// Automatically inject recalled entries.
    pub auto_recall: bool,
    /// Recall result limit.
    pub recall_limit: usize,
    /// Maximum tokens injected per run.
    pub max_context_tokens: usize,
    /// Memory categories read by automatic recall and memory tools.
    pub read_categories: Vec<MemoryCategory>,
    /// Category written by `memory_remember`.
    pub write_category: Option<MemoryCategory>,
}

impl Default for MemorySettings {
    fn default() -> Self {
        Self {
            enabled: false,
            tools: false,
            auto_recall: true,
            recall_limit: 5,
            max_context_tokens: 2_000,
            read_categories: Vec::new(),
            write_category: None,
        }
    }
}

/// Context compression settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompressionSettings {
    /// Compression strategy.
    #[serde(default)]
    pub kind: CompressionKind,
    /// History budget.
    pub max_tokens: usize,
    /// Summary token budget.
    pub summary_tokens: Option<usize>,
}

/// Compression strategy.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressionKind {
    /// Deterministic sliding window.
    #[default]
    Sliding,
    /// Model-generated summary.
    Summarizing,
}

/// Sub-agent tool settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SubagentSettings {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// Nesting limit.
    pub max_depth: usize,
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

/// SKILL.md discovery settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SkillsSettings {
    /// Skill root directories.
    pub paths: Vec<String>,
    /// Directory depth.
    pub max_depth: usize,
}

impl Default for SkillsSettings {
    fn default() -> Self {
        Self {
            paths: Vec::new(),
            max_depth: 4,
        }
    }
}

/// One MCP server.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct McpServerSettings {
    /// Streamable HTTP endpoint. Mutually exclusive with `command`.
    pub url: Option<String>,
    /// stdio executable. Mutually exclusive with `url`.
    pub command: String,
    /// stdio arguments.
    pub args: Vec<String>,
    /// stdio environment.
    pub env: BTreeMap<String, String>,
    /// HTTP headers.
    pub headers: BTreeMap<String, String>,
    /// stdio working directory.
    pub cwd: Option<String>,
    /// Request timeout.
    pub request_timeout_seconds: Option<u64>,
    /// MCP protocol version.
    pub protocol_version: Option<String>,
    /// Client name.
    pub client_name: Option<String>,
    /// Client version.
    pub client_version: Option<String>,
}

impl McpServerSettings {
    fn validate(&self, index: usize) -> Result<(), ContractError> {
        let has_url = self.url.as_ref().is_some_and(|url| !url.trim().is_empty());
        let has_command = !self.command.trim().is_empty();
        if has_url == has_command {
            return Err(invalid(&format!(
                "mcp_servers[{index}] must define exactly one of url or command"
            )));
        }
        if let Some(url) = &self.url
            && !(url.starts_with("http://") || url.starts_with("https://"))
        {
            return Err(invalid(&format!(
                "mcp_servers[{index}].url must use http:// or https://"
            )));
        }
        if self.request_timeout_seconds == Some(0) {
            return Err(invalid(&format!(
                "mcp_servers[{index}].request_timeout_seconds must be greater than zero"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_capabilities() -> CapabilitySet {
        CapabilitySet::new([
            Capability::OpenAi,
            Capability::OpenAiResponses,
            Capability::Anthropic,
            Capability::Gemini,
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

    #[test]
    fn parses_minimal_fixture() {
        let config = AgentConfig::parse_json(include_str!("../fixtures/minimal.json")).unwrap();
        assert_eq!(config.schema_version, SCHEMA_VERSION);
        assert_eq!(config.max_turns, 16);
        config.validate(false, &all_capabilities()).unwrap();
    }

    #[test]
    fn parses_full_fixture() {
        let config = AgentConfig::parse_json(include_str!("../fixtures/full.json")).unwrap();
        config.validate(true, &all_capabilities()).unwrap();
        let required = config.required_capabilities(true);
        assert!(required.contains(Capability::CustomModel));
        assert!(required.contains(Capability::McpStdio));
        assert!(required.contains(Capability::McpHttp));
        assert!(required.contains(Capability::CompressionSummarizing));
        let memory = config.memory_options();
        assert_eq!(memory.read_categories.len(), 3);
        assert_eq!(
            memory.write_category.unwrap(),
            kestrel::MemoryCategory::project("kestrel")
        );
    }

    #[test]
    fn parses_provider_fixture() {
        let config = AgentConfig::parse_json(include_str!("../fixtures/provider.json")).unwrap();
        config.validate(false, &all_capabilities()).unwrap();
        assert_eq!(config.provider_config().kind, ProviderKind::Anthropic);
        assert!(
            config
                .required_capabilities(false)
                .contains(Capability::Anthropic)
        );
    }

    #[test]
    fn rejects_unknown_fields() {
        let error = AgentConfig::parse_json(
            r#"{"schema_version":1,"openai":{"model":"x"},"surprise":true}"#,
        )
        .unwrap_err();
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn rejects_unsupported_capability() {
        let config = AgentConfig::parse_json(
            r#"{
                "schema_version": 1,
                "openai": {"model": "x"},
                "mcp_servers": [{"url": "https://example.com/mcp"}]
            }"#,
        )
        .unwrap();
        let node_capabilities = CapabilitySet::new([Capability::OpenAi, Capability::Streaming]);
        let error = config.validate(false, &node_capabilities).unwrap_err();
        assert!(error.to_string().contains("mcp_http"));
    }

    #[test]
    fn validates_memory_tools() {
        let config = AgentConfig::parse_json(
            r#"{
                "schema_version": 1,
                "openai": {"model": "x"},
                "memory": {"tools": true}
            }"#,
        )
        .unwrap();
        assert!(config.validate(false, &all_capabilities()).is_err());
    }

    #[test]
    fn validates_tool_execution_limits() {
        let mut config = AgentConfig::default();
        config.openai.model = "gpt-4.1-mini".to_owned();
        config.tool_timeout_ms = Some(0);
        assert!(config.validate(false, &all_capabilities()).is_err());

        config.tool_timeout_ms = None;
        config.max_tool_output_chars = Some(0);
        assert!(config.validate(false, &all_capabilities()).is_err());
    }

    #[test]
    fn parses_explicit_provider_config() {
        let config = AgentConfig::parse_json(
            r#"{
                "schema_version": 1,
                "provider": {
                    "kind": "gemini",
                    "model": "gemini-2.5-pro",
                    "api_key": "test"
                }
            }"#,
        )
        .unwrap();
        config.validate(false, &all_capabilities()).unwrap();
        assert!(
            config
                .required_capabilities(false)
                .contains(Capability::Gemini)
        );
        assert_eq!(config.provider_config().kind, ProviderKind::Gemini);
        assert_eq!(config.model_name(), "gemini-2.5-pro");
    }

    #[test]
    fn provider_overrides_legacy_openai_model_requirement() {
        let config = AgentConfig::parse_json(
            r#"{
                "schema_version": 1,
                "openai": {"model": ""},
                "provider": {"kind": "anthropic", "model": "claude-test"}
            }"#,
        )
        .unwrap();
        config.validate(false, &all_capabilities()).unwrap();
        assert_eq!(config.provider_config().kind, ProviderKind::Anthropic);
    }

    #[test]
    fn rejects_provider_without_binding_capability() {
        let config = AgentConfig::parse_json(
            r#"{
                "schema_version": 1,
                "provider": {"kind": "openai_responses", "model": "gpt-test"}
            }"#,
        )
        .unwrap();
        let supported = CapabilitySet::new([Capability::OpenAi, Capability::Streaming]);
        let error = config.validate(false, &supported).unwrap_err();
        assert!(error.to_string().contains("openai_responses"));
    }
}
