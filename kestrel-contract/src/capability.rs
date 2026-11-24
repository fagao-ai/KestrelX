use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// A feature that can be required by configuration or provided by a binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Built-in OpenAI-compatible provider.
    OpenAi,
    /// Host-language custom model callback.
    CustomModel,
    /// Incremental model output.
    Streaming,
    /// Built-in todo tools.
    Todo,
    /// Long-term memory and memory tools.
    Memory,
    /// Deterministic sliding-window compression.
    CompressionSliding,
    /// Model-generated history summarization.
    CompressionSummarizing,
    /// Sub-agent tool.
    Subagent,
    /// SKILL.md discovery.
    Skills,
    /// MCP over stdio.
    McpStdio,
    /// MCP over Streamable HTTP.
    McpHttp,
    /// Host-language tools.
    HostTools,
    /// Host-backed conversation persistence.
    SessionStore,
    /// Host-backed long-term memory.
    MemoryStore,
}

impl Capability {
    /// Stable wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::CustomModel => "custom_model",
            Self::Streaming => "streaming",
            Self::Todo => "todo",
            Self::Memory => "memory",
            Self::CompressionSliding => "compression_sliding",
            Self::CompressionSummarizing => "compression_summarizing",
            Self::Subagent => "subagent",
            Self::Skills => "skills",
            Self::McpStdio => "mcp_stdio",
            Self::McpHttp => "mcp_http",
            Self::HostTools => "host_tools",
            Self::SessionStore => "session_store",
            Self::MemoryStore => "memory_store",
        }
    }
}

impl std::fmt::Display for Capability {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A set of capabilities supported or required by a binding.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CapabilitySet {
    items: BTreeSet<Capability>,
}

impl CapabilitySet {
    /// Creates a capability set.
    pub fn new(capabilities: impl IntoIterator<Item = Capability>) -> Self {
        Self {
            items: capabilities.into_iter().collect(),
        }
    }

    /// Returns whether this set contains a capability.
    pub fn contains(&self, capability: Capability) -> bool {
        self.items.contains(&capability)
    }

    /// Returns the missing subset of `required`.
    pub fn missing(&self, required: &CapabilitySet) -> Vec<Capability> {
        required.items.difference(&self.items).copied().collect()
    }

    /// Returns capabilities in deterministic order.
    pub fn names(&self) -> Vec<&'static str> {
        self.items
            .iter()
            .map(|capability| capability.as_str())
            .collect()
    }

    /// Number of capabilities.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl FromIterator<Capability> for CapabilitySet {
    fn from_iter<T: IntoIterator<Item = Capability>>(iter: T) -> Self {
        Self::new(iter)
    }
}
