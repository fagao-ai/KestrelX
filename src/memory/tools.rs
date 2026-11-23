use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{Memory, MemoryEntry, MemoryQuery, Result, Tool, ToolContext, ToolOutput, ToolSpec};

/// Creates the built-in memory tools: `memory_remember` and `memory_recall`.
pub fn memory_tools(memory: Arc<dyn Memory>) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(RememberTool::new(memory.clone())),
        Arc::new(RecallTool::new(memory)),
    ]
}

/// Tool that writes a long-term memory entry.
pub struct RememberTool {
    memory: Arc<dyn Memory>,
}

impl RememberTool {
    /// Creates the tool.
    pub fn new(memory: Arc<dyn Memory>) -> Self {
        Self { memory }
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

        let entry = MemoryEntry::new(args.text.trim()).with_tags(args.tags);
        let id = entry.id.clone();
        self.memory.remember(entry).await?;
        Ok(ToolOutput::new(format!("Remembered as `{id}`.")))
    }
}

/// Tool that recalls long-term memory entries.
pub struct RecallTool {
    memory: Arc<dyn Memory>,
}

impl RecallTool {
    /// Creates the tool.
    pub fn new(memory: Arc<dyn Memory>) -> Self {
        Self { memory }
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
            .recall(&MemoryQuery::new(args.query).with_limit(limit))
            .await?;
        Ok(ToolOutput::new(serde_json::to_string_pretty(&entries)?))
    }
}
