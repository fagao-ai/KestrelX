use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{AgentBuilder, Error, Result, Tool, ToolContext, ToolOutput, ToolSpec};

/// State of one todo item.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    /// Not started.
    #[default]
    Pending,
    /// Currently being worked on.
    InProgress,
    /// Finished.
    Completed,
}

/// One task in a todo list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TodoItem {
    /// Stable item identifier.
    pub id: String,
    /// Human-readable task.
    pub content: String,
    /// Current state.
    #[serde(default)]
    pub status: TodoStatus,
}

impl TodoItem {
    /// Creates a pending item with a random ID.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.into(),
            status: TodoStatus::Pending,
        }
    }
}

/// A thread-safe todo list shared by its tools.
#[derive(Debug, Clone, Default)]
pub struct TodoList {
    items: Arc<RwLock<Vec<TodoItem>>>,
}

impl TodoList {
    /// Creates an empty list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a list from existing items.
    pub fn from_items(items: impl IntoIterator<Item = TodoItem>) -> Self {
        Self {
            items: Arc::new(RwLock::new(items.into_iter().collect())),
        }
    }

    /// Returns a snapshot of all items.
    pub fn snapshot(&self) -> Vec<TodoItem> {
        self.read().clone()
    }

    /// Replaces every item.
    pub fn replace(&self, items: Vec<TodoItem>) {
        *self.write() = items;
    }

    /// Clears the list.
    pub fn clear(&self) {
        self.write().clear();
    }

    /// Number of items.
    pub fn len(&self) -> usize {
        self.read().len()
    }

    /// Returns whether the list is empty.
    pub fn is_empty(&self) -> bool {
        self.read().is_empty()
    }

    /// Creates `todo_write` and `todo_read` tools.
    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        vec![
            Arc::new(TodoWriteTool::new(self.clone())),
            Arc::new(TodoReadTool::new(self.clone())),
        ]
    }

    /// Adds the todo tools and usage guidance to a builder.
    pub fn install(&self, builder: AgentBuilder) -> AgentBuilder {
        builder
            .tools(self.tools())
            .prompt_section(self.prompt_section())
    }

    /// Returns a prompt section describing how to use the list.
    pub fn prompt_section(&self) -> String {
        "For multi-step work, maintain the todo list with `todo_write`, mark \
         the active item `in_progress`, and keep at most one item in progress. \
         Update it as work advances."
            .to_owned()
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, Vec<TodoItem>> {
        self.items.read().unwrap_or_else(|error| error.into_inner())
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, Vec<TodoItem>> {
        self.items
            .write()
            .unwrap_or_else(|error| error.into_inner())
    }
}

/// Replaces the todo list.
pub struct TodoWriteTool {
    todos: TodoList,
}

impl TodoWriteTool {
    /// Creates the tool.
    pub fn new(todos: TodoList) -> Self {
        Self { todos }
    }
}

#[derive(Deserialize)]
struct WriteArgs {
    todos: Vec<WriteItem>,
}

#[derive(Deserialize)]
struct WriteItem {
    id: Option<String>,
    content: String,
    #[serde(default)]
    status: TodoStatus,
}

#[async_trait]
impl Tool for TodoWriteTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "todo_write",
            "Create or replace the task list. Use for multi-step work. Send the complete desired list, not just changed items.",
            json!({
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "content": { "type": "string" },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"]
                                }
                            },
                            "required": ["content"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["todos"],
                "additionalProperties": false
            }),
        )
    }

    async fn call(&self, arguments: Value, _context: &ToolContext) -> Result<ToolOutput> {
        let args: WriteArgs = serde_json::from_value(arguments)?;
        let mut items = Vec::with_capacity(args.todos.len());
        let mut ids = std::collections::HashSet::new();

        for item in args.todos {
            let content = item.content.trim();
            if content.is_empty() {
                return Ok(ToolOutput::error("todo content cannot be empty"));
            }
            let id = item
                .id
                .filter(|id| !id.trim().is_empty())
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            if !ids.insert(id.clone()) {
                return Err(Error::tool(
                    "todo_write",
                    format!("duplicate todo id `{id}`"),
                ));
            }
            items.push(TodoItem {
                id,
                content: content.to_owned(),
                status: item.status,
            });
        }

        self.todos.replace(items);
        Ok(ToolOutput::new(serde_json::to_string_pretty(
            &self.todos.snapshot(),
        )?))
    }
}

/// Returns the current todo list.
pub struct TodoReadTool {
    todos: TodoList,
}

impl TodoReadTool {
    /// Creates the tool.
    pub fn new(todos: TodoList) -> Self {
        Self { todos }
    }
}

#[async_trait]
impl Tool for TodoReadTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "todo_read",
            "Read the current task list and its statuses.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        )
    }

    async fn call(&self, _arguments: Value, _context: &ToolContext) -> Result<ToolOutput> {
        Ok(ToolOutput::new(serde_json::to_string_pretty(
            &self.todos.snapshot(),
        )?))
    }
}

#[cfg(test)]
mod tests {
    use tokio_util::sync::CancellationToken;

    use super::*;

    #[tokio::test]
    async fn write_and_read_tools_share_state() {
        let todos = TodoList::new();
        let context = ToolContext::new("call", "session", 0, CancellationToken::new());
        let write = TodoWriteTool::new(todos.clone());
        write
            .call(
                json!({
                    "todos": [
                        { "content": "Draft", "status": "completed" },
                        { "content": "Ship", "status": "in_progress" }
                    ]
                }),
                &context,
            )
            .await
            .unwrap();

        assert_eq!(todos.len(), 2);
        let output = TodoReadTool::new(todos)
            .call(json!({}), &context)
            .await
            .unwrap();
        assert!(output.content.contains("Ship"));
        assert!(output.content.contains("in_progress"));
    }
}
