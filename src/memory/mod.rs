mod store;
mod tools;

use std::collections::BTreeMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Result, util::now_ms};

pub use store::InMemoryMemory;
pub use tools::{RecallTool, RememberTool, memory_tools};

/// A durable fact or note available across runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Stable entry identifier.
    pub id: String,
    /// Memory text shown to the model.
    pub text: String,
    /// Optional retrieval tags.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Application-owned metadata.
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
    /// Creation time in Unix milliseconds.
    #[serde(default)]
    pub created_at_ms: u64,
}

impl MemoryEntry {
    /// Creates an entry with a random ID and current timestamp.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            text: text.into(),
            tags: Vec::new(),
            metadata: BTreeMap::new(),
            created_at_ms: now_ms(),
        }
    }

    /// Adds retrieval tags.
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags = tags.into_iter().map(Into::into).collect();
        self
    }
}

/// A long-term memory query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQuery {
    /// Free-form retrieval query.
    pub text: String,
    /// Optional exact tags.
    pub tags: Vec<String>,
    /// Maximum number of entries.
    pub limit: usize,
}

impl MemoryQuery {
    /// Creates a query with a default limit of five.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tags: Vec::new(),
            limit: 5,
        }
    }

    /// Sets the result limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

/// Pluggable long-term memory.
///
/// Retrieval quality is entirely up to the implementation. An embedding or
/// vector-database adapter can be dropped in without changing the agent loop.
#[async_trait]
pub trait Memory: Send + Sync {
    /// Stores or replaces one entry.
    async fn remember(&self, entry: MemoryEntry) -> Result<()>;

    /// Retrieves relevant entries.
    async fn recall(&self, query: &MemoryQuery) -> Result<Vec<MemoryEntry>>;
}

/// Options for automatic memory recall.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryOptions {
    /// Inject recalled entries before each run.
    pub auto_recall: bool,
    /// Maximum number of recalled entries.
    pub recall_limit: usize,
    /// Maximum tokens spent on the injected memory block.
    pub max_context_tokens: usize,
}

impl Default for MemoryOptions {
    fn default() -> Self {
        Self {
            auto_recall: true,
            recall_limit: 5,
            max_context_tokens: 2_000,
        }
    }
}
