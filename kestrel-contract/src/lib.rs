//! Versioned configuration and capability contract shared by Kestrel SDKs.
//!
//! This crate is deliberately provider-neutral. Bindings use it to parse the
//! same JSON shape and reject unsupported capabilities before constructing an
//! agent.

mod capability;
mod config;
mod error;

pub use capability::{Capability, CapabilitySet};
pub use config::{
    AgentConfig, CompressionKind, CompressionSettings, McpServerSettings, MemorySettings,
    OpenAiSettings, SkillsSettings, SubagentSettings,
};
pub use error::ContractError;

/// Current configuration schema version.
pub const SCHEMA_VERSION: u32 = 1;
