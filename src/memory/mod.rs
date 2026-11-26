mod store;
mod tools;

use std::collections::BTreeMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Result, util::now_ms};

pub use store::InMemoryMemory;
pub use tools::{RecallTool, RememberTool, memory_tools, memory_tools_with_options};

/// A memory namespace category.
///
/// `name` is application-defined, for example `user`, `project`, `session`, or
/// `global`. `owner_id` is the corresponding stable identifier, if any.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryCategory {
    /// Category name.
    pub name: String,
    /// Application-defined owner identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owner_id: Option<String>,
}

impl MemoryCategory {
    /// Creates a category.
    pub fn new(name: impl Into<String>, owner_id: Option<impl Into<String>>) -> Self {
        Self {
            name: name.into(),
            owner_id: owner_id.map(Into::into),
        }
    }

    /// Creates a category without an owner.
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            owner_id: None,
        }
    }

    /// Creates a user-scoped category.
    pub fn user(user_id: impl Into<String>) -> Self {
        Self::new("user", Some(user_id))
    }

    /// Creates a project-scoped category.
    pub fn project(project_id: impl Into<String>) -> Self {
        Self::new("project", Some(project_id))
    }

    /// Creates a global category.
    pub fn global() -> Self {
        Self::named("global")
    }

    /// Returns a human-readable label for prompt injection.
    pub fn label(&self) -> String {
        match &self.owner_id {
            Some(owner_id) if self.name == "default" => owner_id.clone(),
            Some(owner_id) => format!("{}:{owner_id}", self.name),
            None => self.name.clone(),
        }
    }

    /// Returns whether this is the fallback category.
    pub fn is_default(&self) -> bool {
        self.name == "default" && self.owner_id.is_none()
    }
}

impl Default for MemoryCategory {
    fn default() -> Self {
        Self::named("default")
    }
}

/// A durable fact or note available across runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Stable entry identifier.
    pub id: String,
    /// Memory category and owner namespace.
    #[serde(default)]
    pub category: MemoryCategory,
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
            category: MemoryCategory::default(),
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

    /// Sets the memory category.
    pub fn with_category(mut self, category: MemoryCategory) -> Self {
        self.category = category;
        self
    }
}

/// A long-term memory query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQuery {
    /// Categories to search, ordered from highest to lowest priority.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub categories: Vec<MemoryCategory>,
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
            categories: Vec::new(),
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

    /// Restricts recall to the supplied categories.
    pub fn with_categories(mut self, categories: impl IntoIterator<Item = MemoryCategory>) -> Self {
        self.categories = categories.into_iter().collect();
        self
    }

    /// Adds one category to the recall filter.
    pub fn with_category(mut self, category: MemoryCategory) -> Self {
        self.categories.push(category);
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
    /// Categories searched by automatic and tool-based recall.
    #[serde(default)]
    pub read_categories: Vec<MemoryCategory>,
    /// Category written by the built-in `memory_remember` tool.
    #[serde(default)]
    pub write_category: Option<MemoryCategory>,
}

impl Default for MemoryOptions {
    fn default() -> Self {
        Self {
            auto_recall: true,
            recall_limit: 5,
            max_context_tokens: 2_000,
            read_categories: Vec::new(),
            write_category: None,
        }
    }
}
