//! Optional Model Context Protocol integration over stdio or Streamable HTTP.
//!
//! This module only adapts MCP tools to the `Tool` trait. It is feature-gated
//! and has no presence in the core dependency graph unless enabled.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::Mutex,
    time::timeout,
};

use crate::{Error, Result, Tool, ToolContext, ToolOutput, ToolSpec};

mod http;

pub use http::{McpHttpClient, McpHttpConfig};

/// Configuration for one MCP server process.
#[derive(Clone)]
pub struct McpServerConfig {
    /// Executable to launch.
    pub command: String,
    /// Command-line arguments.
    pub args: Vec<String>,
    /// Extra environment variables.
    pub env: BTreeMap<String, String>,
    /// Working directory.
    pub cwd: Option<PathBuf>,
    /// Timeout for each JSON-RPC request.
    pub request_timeout: Duration,
    /// MCP protocol version sent during initialization.
    pub protocol_version: String,
    /// Client name reported to the server.
    pub client_name: String,
    /// Client version reported to the server.
    pub client_version: String,
}

impl McpServerConfig {
    /// Creates stdio configuration for a command.
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            args: Vec::new(),
            env: BTreeMap::new(),
            cwd: None,
            request_timeout: Duration::from_secs(30),
            protocol_version: "2025-06-18".to_owned(),
            client_name: "kestrel".to_owned(),
            client_version: env!("CARGO_PKG_VERSION").to_owned(),
        }
    }

    /// Adds one command-line argument.
    pub fn arg(mut self, argument: impl Into<String>) -> Self {
        self.args.push(argument.into());
        self
    }

    /// Adds multiple command-line arguments.
    pub fn args(mut self, arguments: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.args.extend(arguments.into_iter().map(Into::into));
        self
    }

    /// Adds one environment variable.
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// Sets the working directory.
    pub fn cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Sets the JSON-RPC request timeout.
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Sets the MCP protocol version.
    pub fn protocol_version(mut self, version: impl Into<String>) -> Self {
        self.protocol_version = version.into();
        self
    }
}

impl std::fmt::Debug for McpServerConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("McpServerConfig")
            .field("command", &self.command)
            .field("args", &self.args)
            .field("env_keys", &self.env.keys().collect::<Vec<_>>())
            .field("cwd", &self.cwd)
            .field("request_timeout", &self.request_timeout)
            .field("protocol_version", &self.protocol_version)
            .finish()
    }
}

/// A connected stdio MCP server.
#[derive(Clone)]
pub struct McpClient {
    inner: Arc<McpInner>,
}

struct McpInner {
    config: McpServerConfig,
    transport: Mutex<StdioTransport>,
    next_id: AtomicU64,
}

struct StdioTransport {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl std::fmt::Debug for McpClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("McpClient")
            .field("config", &self.inner.config)
            .finish_non_exhaustive()
    }
}

impl McpClient {
    /// Launches a server process and performs the MCP initialization handshake.
    pub async fn connect(config: McpServerConfig) -> Result<Self> {
        let mut command = Command::new(&config.command);
        command
            .args(&config.args)
            .envs(&config.env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if let Some(cwd) = &config.cwd {
            command.current_dir(cwd);
        }

        let mut child = command.spawn().map_err(|error| {
            Error::Configuration(format!(
                "failed to launch MCP server `{}`: {error}",
                config.command
            ))
        })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| Error::Protocol("MCP server stdin was not available".to_owned()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::Protocol("MCP server stdout was not available".to_owned()))?;

        let client = Self {
            inner: Arc::new(McpInner {
                config,
                transport: Mutex::new(StdioTransport {
                    child,
                    stdin: BufWriter::new(stdin),
                    stdout: BufReader::new(stdout),
                }),
                next_id: AtomicU64::new(1),
            }),
        };
        if let Err(error) = client.initialize().await {
            let _ = client.close().await;
            return Err(error);
        }
        Ok(client)
    }

    /// Returns the server configuration.
    pub fn config(&self) -> &McpServerConfig {
        &self.inner.config
    }

    /// Discovers all tools, following pagination.
    pub async fn tools(&self) -> Result<Vec<Arc<dyn Tool>>> {
        let mut discovered = Vec::new();
        let mut names = BTreeSet::new();
        let mut cursor: Option<String> = None;

        for page_number in 0..100 {
            let params = cursor
                .as_ref()
                .map(|cursor| json!({ "cursor": cursor }))
                .unwrap_or_else(|| json!({}));
            let result = self.request("tools/list", params).await?;
            let page: ListToolsResult = serde_json::from_value(result)?;
            for remote in page.tools {
                let exposed_name = sanitize_tool_name(&remote.name);
                if !names.insert(exposed_name.clone()) {
                    return Err(Error::Protocol(format!(
                        "MCP server returned colliding tool name `{}` (exposed as `{exposed_name}`)",
                        remote.name,
                    )));
                }
                let spec = ToolSpec::new(
                    exposed_name,
                    remote.description.unwrap_or_default(),
                    remote
                        .input_schema
                        .unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
                );
                discovered.push(Arc::new(McpTool {
                    client: self.clone(),
                    remote_name: remote.name,
                    spec,
                }) as Arc<dyn Tool>);
            }
            match page.next_cursor {
                Some(next) if !next.is_empty() => {
                    if page_number == 99 {
                        return Err(Error::Protocol(
                            "MCP tools/list exceeded 100 pages".to_owned(),
                        ));
                    }
                    cursor = Some(next);
                }
                _ => break,
            }
        }

        Ok(discovered)
    }

    /// Calls one MCP tool directly.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<ToolOutput> {
        let result = self
            .request(
                "tools/call",
                json!({
                    "name": name,
                    "arguments": arguments
                }),
            )
            .await?;
        Ok(render_tool_result(&result))
    }

    /// Stops the server process.
    pub async fn close(&self) -> Result<()> {
        let mut transport = self.inner.transport.lock().await;
        let _ = transport.child.start_kill();
        transport.child.wait().await?;
        Ok(())
    }

    async fn initialize(&self) -> Result<()> {
        let result = self
            .request(
                "initialize",
                json!({
                    "protocolVersion": self.inner.config.protocol_version,
                    "capabilities": {},
                    "clientInfo": {
                        "name": self.inner.config.client_name,
                        "version": self.inner.config.client_version
                    }
                }),
            )
            .await?;
        if !result.is_object() {
            return Err(Error::Protocol(
                "initialize result must be an object".to_owned(),
            ));
        }
        self.notify("notifications/initialized", json!({})).await
    }

    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let message = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });
        let mut transport = self.inner.transport.lock().await;
        write_json(&mut transport, &message).await
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let request = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        });
        let mut transport = self.inner.transport.lock().await;

        let result = timeout(self.inner.config.request_timeout, async {
            write_json(&mut transport, &request).await?;
            read_response(&mut transport, id).await
        })
        .await;

        match result {
            Ok(response) => response,
            Err(_) => Err(Error::Protocol(format!(
                "MCP request `{method}` timed out after {:?}",
                self.inner.config.request_timeout
            ))),
        }
    }
}

impl Drop for McpInner {
    fn drop(&mut self) {
        if let Ok(mut transport) = self.transport.try_lock() {
            let _ = transport.child.start_kill();
        }
    }
}

/// One remote MCP tool.
pub struct McpTool {
    client: McpClient,
    remote_name: String,
    spec: ToolSpec,
}

impl std::fmt::Debug for McpTool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("McpTool")
            .field("spec", &self.spec)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Tool for McpTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, arguments: Value, _context: &ToolContext) -> Result<ToolOutput> {
        self.client.call_tool(&self.remote_name, arguments).await
    }
}

async fn write_json(transport: &mut StdioTransport, value: &Value) -> Result<()> {
    let line = serde_json::to_string(value)?;
    transport.stdin.write_all(line.as_bytes()).await?;
    transport.stdin.write_all(b"\n").await?;
    transport.stdin.flush().await?;
    Ok(())
}

async fn read_response(transport: &mut StdioTransport, id: u64) -> Result<Value> {
    let expected = Value::from(id);
    let mut line = String::new();

    loop {
        line.clear();
        let bytes = transport.stdout.read_line(&mut line).await?;
        if bytes == 0 {
            return Err(Error::Protocol(
                "MCP server closed stdout before responding".to_owned(),
            ));
        }
        if line.trim().is_empty() {
            continue;
        }

        let message: Value = serde_json::from_str(&line)
            .map_err(|error| Error::Protocol(format!("invalid MCP JSON: {error}")))?;
        if message.get("method").is_some() && message.get("id").is_some() {
            let method = message
                .get("method")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let response = if method == "ping" {
                json!({
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {}
                })
            } else {
                json!({
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {
                        "code": -32601,
                        "message": "client does not implement this server request"
                    }
                })
            };
            write_json(transport, &response).await?;
            continue;
        }
        if message.get("id") != Some(&expected) {
            continue;
        }
        if let Some(error) = message.get("error") {
            let code = error
                .get("code")
                .and_then(Value::as_i64)
                .unwrap_or_default();
            let description = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown MCP error");
            return Err(Error::Protocol(format!(
                "JSON-RPC error {code}: {description}"
            )));
        }
        return message.get("result").cloned().ok_or_else(|| {
            Error::Protocol("MCP response contains neither result nor error".to_owned())
        });
    }
}

pub(super) fn render_tool_result(result: &Value) -> ToolOutput {
    let is_error = result
        .get("isError")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut content = Vec::new();

    if let Some(items) = result.get("content").and_then(Value::as_array) {
        for item in items {
            match item.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(text) = item.get("text").and_then(Value::as_str) {
                        content.push(text.to_owned());
                    }
                }
                Some("image") => {
                    let mime = item
                        .get("mimeType")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown");
                    content.push(format!("[image: {mime}]"));
                }
                Some("audio") => {
                    let mime = item
                        .get("mimeType")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown");
                    content.push(format!("[audio: {mime}]"));
                }
                Some("resource") => {
                    let resource = item.get("resource").unwrap_or(item);
                    if let Some(text) = resource.get("text").and_then(Value::as_str) {
                        content.push(text.to_owned());
                    } else if let Some(uri) = resource.get("uri").and_then(Value::as_str) {
                        content.push(format!("[resource: {uri}]"));
                    }
                }
                _ => content.push(item.to_string()),
            }
        }
    }

    if content.is_empty() {
        if let Some(structured) = result.get("structuredContent") {
            content.push(
                serde_json::to_string_pretty(structured).unwrap_or_else(|_| structured.to_string()),
            );
        } else {
            content.push(result.to_string());
        }
    }

    ToolOutput {
        content: content.join("\n"),
        is_error,
    }
}

pub(super) fn sanitize_tool_name(name: &str) -> String {
    let mut sanitized = name
        .chars()
        .take(64)
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        sanitized.push_str("mcp_tool");
    }
    sanitized
}

#[derive(Deserialize)]
pub(super) struct ListToolsResult {
    #[serde(default)]
    pub(super) tools: Vec<RemoteTool>,
    #[serde(default, rename = "nextCursor")]
    pub(super) next_cursor: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct RemoteTool {
    pub(super) name: String,
    pub(super) description: Option<String>,
    #[serde(default, rename = "inputSchema")]
    pub(super) input_schema: Option<Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_text_and_flags_errors() {
        let output = render_tool_result(&json!({
            "content": [{ "type": "text", "text": "failed" }],
            "isError": true
        }));
        assert_eq!(output.content, "failed");
        assert!(output.is_error);
    }

    #[test]
    fn falls_back_to_structured_content() {
        let output = render_tool_result(&json!({
            "content": [],
            "structuredContent": { "answer": 42 }
        }));
        assert!(output.content.contains("\"answer\": 42"));
    }

    #[test]
    fn sanitizes_mcp_tool_names() {
        assert_eq!(sanitize_tool_name("repo.search"), "repo_search");
        assert_eq!(sanitize_tool_name(""), "mcp_tool");
    }
}
