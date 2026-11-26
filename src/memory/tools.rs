use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    Memory, MemoryCategory, MemoryEntry, MemoryOptions, MemoryQuery, Result, Tool, ToolContext,
    ToolOutput, ToolSpec,
};

/// Creates the built-in memory tools: `memory_remember` and `memory_recall`.
pub fn memory_tools(memory: Arc<dyn Memory>) -> Vec<Arc<dyn Tool>> {
    memory_tools_with_options(memory, MemoryOptions::default())
}

/// Creates memory tools using explicit category read/write options.
pub fn memory_tools_with_options(
    memory: Arc<dyn Memory>,
    options: MemoryOptions,
) -> Vec<Arc<dyn Tool>> {
    let write_category = options.write_category.unwrap_or_default();
    vec![
        Arc::new(RememberTool::new(memory.clone()).with_category(write_category)),
        Arc::new(RecallTool::new(memory).with_categories(options.read_categories)),
    ]
}

/// Tool that writes a long-term memory entry.
pub struct RememberTool {
    memory: Arc<dyn Memory>,
    category: MemoryCategory,
}

impl RememberTool {
    /// Creates the tool.
    pub fn new(memory: Arc<dyn Memory>) -> Self {
        Self {
            memory,
            category: MemoryCategory::default(),
        }
    }

    /// Sets the category written by this tool.
    pub fn with_category(mut self, category: MemoryCategory) -> Self {
        self.category = category;
        self
    }
}

#[derive(Deserialize)]
struct RememberArgs {
    text: String,
    #[serde(default)]
    tags: Vec<String>,
}

#[async_trait]
impl Tool for RememberTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "memory_remember",
            "Store a durable fact, preference, or decision for future runs. Use only for information that will remain useful.",
            json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The standalone fact or note to remember."
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional retrieval tags."
                    }
                },
                "required": ["text"],
                "additionalProperties": false
            }),
        )
    }

    async fn call(&self, arguments: Value, _context: &ToolContext) -> Result<ToolOutput> {
        let args: RememberArgs = serde_json::from_value(arguments)?;
        if args.text.trim().is_empty() {
            return Ok(ToolOutput::error("memory text cannot be empty"));
        }

        let entry = MemoryEntry::new(args.text.trim())
            .with_category(self.category.clone())
            .with_tags(args.tags);
        let id = entry.id.clone();
        self.memory.remember(entry).await?;
        Ok(ToolOutput::new(format!("Remembered as `{id}`.")))
    }
}

/// Tool that recalls long-term memory entries.
pub struct RecallTool {
    memory: Arc<dyn Memory>,
    categories: Vec<MemoryCategory>,
}

impl RecallTool {
    /// Creates the tool.
    pub fn new(memory: Arc<dyn Memory>) -> Self {
        Self {
            memory,
            categories: Vec::new(),
        }
    }

    /// Restricts recall to the supplied categories.
    pub fn with_categories(mut self, categories: Vec<MemoryCategory>) -> Self {
        self.categories = categories;
        self
    }
}

#[derive(Deserialize)]
struct RecallArgs {
    query: String,
    limit: Option<usize>,
}

#[async_trait]
impl Tool for RecallTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "memory_recall",
            "Search durable memory for facts relevant to the current task.",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        )
    }

    async fn call(&self, arguments: Value, _context: &ToolContext) -> Result<ToolOutput> {
        let args: RecallArgs = serde_json::from_value(arguments)?;
        let limit = args.limit.unwrap_or(5).clamp(1, 20);
        let entries = self
            .memory
            .recall(
                &MemoryQuery::new(args.query)
                    .with_limit(limit)
                    .with_categories(self.categories.clone()),
            )
            .await?;
        Ok(ToolOutput::new(serde_json::to_string_pretty(&entries)?))
    }
}

#[cfg(test)]
mod tests {
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::InMemoryMemory;

    #[tokio::test]
    async fn tools_use_configured_categories() {
        let memory = Arc::new(InMemoryMemory::new());
        let options = MemoryOptions {
            read_categories: vec![MemoryCategory::project("repo")],
            write_category: Some(MemoryCategory::project("repo")),
            ..Default::default()
        };
        let tools = memory_tools_with_options(memory.clone(), options);
        let remember = tools
            .iter()
            .find(|tool| tool.spec().name == "memory_remember")
            .unwrap();
        let recall = tools
            .iter()
            .find(|tool| tool.spec().name == "memory_recall")
            .unwrap();
        let context = ToolContext::new("call", "session", 0, CancellationToken::new());

        remember
            .call(json!({ "text": "Use the project convention" }), &context)
            .await
            .unwrap();
        let project_entries = memory
            .recall(
                &MemoryQuery::new("convention").with_categories([MemoryCategory::project("repo")]),
            )
            .await
            .unwrap();
        assert_eq!(project_entries.len(), 1);
        assert_eq!(project_entries[0].category.name, "project");

        let user_entries = memory
            .recall(
                &MemoryQuery::new("convention").with_categories([MemoryCategory::user("alice")]),
            )
            .await
            .unwrap();
        assert!(user_entries.is_empty());

        let output = recall
            .call(json!({ "query": "convention" }), &context)
            .await
            .unwrap();
        assert!(output.content.contains("project"));
        assert!(output.content.contains("Use the project convention"));
    }
}
