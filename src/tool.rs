use std::{
    collections::{BTreeMap, HashSet},
    future::Future,
    sync::{Arc, RwLock},
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::{Error, Result};

/// How tool calls from one assistant message are executed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolConcurrency {
    /// Execute tool calls in request order.
    #[default]
    Serial,
    /// Poll all requested tool calls concurrently.
    Parallel,
}

/// JSON Schema description of a tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Name used by the model.
    pub name: String,
    /// Natural-language guidance for the model.
    pub description: String,
    /// JSON Schema for the arguments object.
    pub input_schema: Value,
}

impl ToolSpec {
    /// Creates a tool specification.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

/// Result returned to the model after a tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolOutput {
    /// Text shown to the model.
    pub content: String,
    /// Whether the tool failed. Failed outputs are normally fed back so the
    /// model can recover instead of aborting the run.
    #[serde(default, skip_serializing_if = "is_false")]
    pub is_error: bool,
}

impl ToolOutput {
    /// Creates a successful output.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    /// Creates an output that represents a recoverable tool error.
    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// Runtime information available to tools.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Provider-assigned tool call ID.
    pub call_id: String,
    /// Current session ID.
    pub session_id: String,
    /// Agent nesting depth. Root agents start at zero.
    pub depth: usize,
    cancellation: CancellationToken,
}

impl ToolContext {
    /// Creates a tool context. This is mainly useful in tests and custom
    /// runtime integrations.
    pub fn new(
        call_id: impl Into<String>,
        session_id: impl Into<String>,
        depth: usize,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            session_id: session_id.into(),
            depth,
            cancellation,
        }
    }

    /// Returns a cancellation token shared with the current run.
    pub fn cancellation(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    /// Returns whether the current run has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
}

/// A capability exposed to a model.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Describes the tool to the model.
    fn spec(&self) -> ToolSpec;

    /// Executes the tool with decoded JSON arguments.
    async fn call(&self, arguments: Value, context: &ToolContext) -> Result<ToolOutput>;
}

impl std::fmt::Debug for dyn Tool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Tool")
            .field("spec", &self.spec())
            .finish_non_exhaustive()
    }
}

struct FunctionTool<F> {
    spec: ToolSpec,
    function: F,
}

#[async_trait]
impl<F, Fut> Tool for FunctionTool<F>
where
    F: Fn(Value, ToolContext) -> Fut + Send + Sync,
    Fut: Future<Output = Result<ToolOutput>> + Send,
{
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, arguments: Value, context: &ToolContext) -> Result<ToolOutput> {
        let future = (self.function)(arguments, context.clone());
        future.await
    }
}

/// Creates a tool from an async function.
///
/// ```
/// use kestrel::{function_tool, ToolOutput, ToolSpec};
/// use serde_json::json;
///
/// let echo = function_tool(
///     ToolSpec::new("echo", "Echo text", json!({
///         "type": "object",
///         "properties": { "text": { "type": "string" } },
///         "required": ["text"]
///     })),
///     |args, _context| async move {
///         let text = args["text"].as_str().unwrap_or_default();
///         Ok(ToolOutput::new(text))
///     },
/// );
/// ```
pub fn function_tool<F, Fut>(spec: ToolSpec, function: F) -> Arc<dyn Tool>
where
    F: Fn(Value, ToolContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<ToolOutput>> + Send + 'static,
{
    Arc::new(FunctionTool { spec, function })
}

/// A thread-safe, dynamically extensible tool collection.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: Arc<RwLock<BTreeMap<String, Arc<dyn Tool>>>>,
}

impl ToolRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers one tool.
    ///
    /// Registration is rejected when the name is empty or already present.
    pub fn register(&self, tool: Arc<dyn Tool>) -> Result<()> {
        let spec = tool.spec();
        validate_tool_name(&spec.name)?;
        let mut tools = self.write();
        if tools.contains_key(&spec.name) {
            return Err(Error::Configuration(format!(
                "tool `{}` is already registered",
                spec.name
            )));
        }
        tools.insert(spec.name, tool);
        Ok(())
    }

    /// Registers all tools from an iterator.
    pub fn register_all(&self, tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> Result<()> {
        let mut entries = Vec::new();
        for tool in tools {
            let spec = tool.spec();
            validate_tool_name(&spec.name)?;
            entries.push((spec.name, tool));
        }

        let mut registry = self.write();
        let mut names = HashSet::new();
        for (name, _) in &entries {
            if registry.contains_key(name) || !names.insert(name.clone()) {
                return Err(Error::Configuration(format!(
                    "tool `{name}` is already registered"
                )));
            }
        }
        for (name, tool) in entries {
            registry.insert(name, tool);
        }
        Ok(())
    }

    /// Replaces an existing tool, or inserts it when absent.
    pub fn register_or_replace(&self, tool: Arc<dyn Tool>) -> Result<()> {
        let spec = tool.spec();
        validate_tool_name(&spec.name)?;
        self.write().insert(spec.name, tool);
        Ok(())
    }

    /// Removes a tool by name.
    pub fn remove(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.write().remove(name)
    }

    /// Looks up a tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.read().get(name).cloned()
    }

    /// Returns whether a tool is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.read().contains_key(name)
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.read().len()
    }

    /// Returns whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.read().is_empty()
    }

    /// Returns specifications in deterministic name order.
    pub fn specs(&self) -> Vec<ToolSpec> {
        let tools = self.read().values().cloned().collect::<Vec<_>>();
        tools.iter().map(|tool| tool.spec()).collect()
    }

    /// Returns all registered tools.
    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.read().values().cloned().collect()
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, BTreeMap<String, Arc<dyn Tool>>> {
        self.tools.read().unwrap_or_else(|error| error.into_inner())
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, BTreeMap<String, Arc<dyn Tool>>> {
        self.tools
            .write()
            .unwrap_or_else(|error| error.into_inner())
    }
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ToolRegistry")
            .field("tools", &self.specs())
            .finish()
    }
}

fn validate_tool_name(name: &str) -> Result<()> {
    let valid = !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'));
    if valid {
        Ok(())
    } else {
        Err(Error::Configuration(format!(
            "invalid tool name `{name}`: use 1-64 ASCII letters, digits, `_`, or `-`"
        )))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn test_tool(name: &str) -> Arc<dyn Tool> {
        function_tool(
            ToolSpec::new(name, "test", json!({ "type": "object" })),
            |_arguments, _context| async { Ok(ToolOutput::new("ok")) },
        )
    }

    #[test]
    fn batch_registration_is_atomic() {
        let registry = ToolRegistry::new();
        registry.register(test_tool("existing")).unwrap();

        let result = registry.register_all([test_tool("new"), test_tool("existing")]);
        assert!(result.is_err());
        assert!(!registry.contains("new"));
    }
}
