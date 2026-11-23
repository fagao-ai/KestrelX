use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, RwLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use reqwest::{
    Client, Response, Url,
    header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue},
};
use serde_json::{Value, json};
use tokio::time::timeout;

use crate::{Error, Result, Tool, ToolContext, ToolOutput, ToolSpec};

use super::{ListToolsResult, render_tool_result, sanitize_tool_name};

/// Configuration for one Streamable HTTP MCP endpoint.
#[derive(Clone)]
pub struct McpHttpConfig {
    /// MCP endpoint URL.
    pub url: String,
    /// Extra request headers, including authorization if needed.
    pub headers: BTreeMap<String, String>,
    /// Timeout for one JSON-RPC request.
    pub request_timeout: Duration,
    /// MCP protocol version sent during initialization.
    pub protocol_version: String,
    /// Client name reported to the server.
    pub client_name: String,
    /// Client version reported to the server.
    pub client_version: String,
}

impl McpHttpConfig {
    /// Creates configuration for an endpoint.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            headers: BTreeMap::new(),
            request_timeout: Duration::from_secs(30),
            protocol_version: "2025-06-18".to_owned(),
            client_name: "kestrel".to_owned(),
            client_version: env!("CARGO_PKG_VERSION").to_owned(),
        }
    }

    /// Adds one request header.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
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

impl std::fmt::Debug for McpHttpConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("McpHttpConfig")
            .field("url", &self.url)
            .field("header_names", &self.headers.keys().collect::<Vec<_>>())
            .field("request_timeout", &self.request_timeout)
            .field("protocol_version", &self.protocol_version)
            .finish()
    }
}

/// A Streamable HTTP MCP client.
#[derive(Clone)]
pub struct McpHttpClient {
    inner: Arc<HttpInner>,
}

struct HttpInner {
    client: Client,
    url: Url,
    headers: HeaderMap,
    request_timeout: Duration,
    protocol_version: String,
    client_name: String,
    client_version: String,
    next_id: AtomicU64,
    session_id: RwLock<Option<String>>,
    negotiated_protocol: RwLock<Option<String>>,
}

impl std::fmt::Debug for McpHttpClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("McpHttpClient")
            .field("url", &self.inner.url)
            .field("has_session", &self.session_id().is_some())
            .finish_non_exhaustive()
    }
}

impl McpHttpClient {
    /// Connects and performs the MCP initialization handshake.
    pub async fn connect(config: McpHttpConfig) -> Result<Self> {
        let url = Url::parse(&config.url).map_err(|error| {
            Error::Configuration(format!("invalid MCP HTTP URL `{}`: {error}", config.url))
        })?;
        let mut headers = HeaderMap::new();
        for (name, value) in &config.headers {
            let name = HeaderName::from_bytes(name.as_bytes()).map_err(|error| {
                Error::Configuration(format!("invalid MCP HTTP header name `{name}`: {error}"))
            })?;
            let value = HeaderValue::from_str(value).map_err(|error| {
                Error::Configuration(format!("invalid MCP HTTP header value: {error}"))
            })?;
            headers.insert(name, value);
        }
        let client = Client::builder()
            .user_agent(format!("{}/{}", config.client_name, config.client_version))
            .build()
            .map_err(Error::Http)?;
        let client = Self {
            inner: Arc::new(HttpInner {
                client,
                url,
                headers,
                request_timeout: config.request_timeout.max(Duration::from_millis(1)),
                protocol_version: config.protocol_version,
                client_name: config.client_name,
                client_version: config.client_version,
                next_id: AtomicU64::new(1),
                session_id: RwLock::new(None),
                negotiated_protocol: RwLock::new(None),
            }),
        };
        if let Err(error) = client.initialize().await {
            let _ = client.close().await;
            return Err(error);
        }
        Ok(client)
    }

    /// Returns the server-assigned session ID, if any.
    pub fn session_id(&self) -> Option<String> {
        self.inner
            .session_id
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
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
                discovered.push(Arc::new(McpHttpTool {
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

    /// Terminates the server session when supported.
    pub async fn close(&self) -> Result<()> {
        let Some(session_id) = self
            .inner
            .session_id
            .write()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        else {
            return Ok(());
        };
        let mut headers = self.base_headers();
        if let Ok(value) = HeaderValue::from_str(&session_id) {
            headers.insert(HeaderName::from_static("mcp-session-id"), value);
        }
        let response = self
            .inner
            .client
            .delete(self.inner.url.clone())
            .headers(headers)
            .send()
            .await;
        if let Ok(response) = response {
            let _ = response.bytes().await;
        }
        Ok(())
    }

    async fn initialize(&self) -> Result<()> {
        let result = self
            .request(
                "initialize",
                json!({
                    "protocolVersion": self.inner.protocol_version,
                    "capabilities": {},
                    "clientInfo": {
                        "name": self.inner.client_name,
                        "version": self.inner.client_version
                    }
                }),
            )
            .await?;
        if !result.is_object() {
            return Err(Error::Protocol(
                "initialize result must be an object".to_owned(),
            ));
        }
        if let Some(version) = result.get("protocolVersion").and_then(Value::as_str) {
            *self
                .inner
                .negotiated_protocol
                .write()
                .unwrap_or_else(|error| error.into_inner()) = Some(version.to_owned());
        }
        self.notify("notifications/initialized", json!({})).await
    }

    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let message = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });
        let response = timeout(self.inner.request_timeout, self.post(&message))
            .await
            .map_err(|_| {
                Error::Protocol(format!("MCP HTTP notification `{method}` timed out"))
            })??;
        self.capture_session(response.headers())?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(http_status_error(status.as_u16(), &body));
        }
        Ok(())
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let message = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        });
        timeout(
            self.inner.request_timeout,
            self.request_inner(id, &message, method),
        )
        .await
        .map_err(|_| {
            Error::Protocol(format!(
                "MCP HTTP request `{method}` timed out after {:?}",
                self.inner.request_timeout
            ))
        })?
    }

    async fn request_inner(&self, id: u64, message: &Value, method: &str) -> Result<Value> {
        let response = self.post(message).await?;
        self.capture_session(response.headers())?;
        self.capture_protocol(response.headers())?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            if status.as_u16() == 404 {
                *self
                    .inner
                    .session_id
                    .write()
                    .unwrap_or_else(|error| error.into_inner()) = None;
            }
            return Err(http_status_error(status.as_u16(), &body));
        }
        let is_sse = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(|value| value.to_ascii_lowercase().contains("text/event-stream"))
            .unwrap_or(false);

        if is_sse {
            self.read_sse_response(response, id, method).await
        } else {
            let bytes = response.bytes().await.map_err(Error::Http)?;
            let value: Value = serde_json::from_slice(&bytes)?;
            extract_response(&value, id)?.ok_or_else(|| {
                Error::Protocol(format!(
                    "MCP HTTP response for `{method}` did not contain id {id}"
                ))
            })
        }
    }

    async fn post(&self, message: &Value) -> Result<Response> {
        let mut headers = self.base_headers();
        headers.insert(
            ACCEPT,
            HeaderValue::from_static("application/json, text/event-stream"),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(session_id) = self.session_id() {
            let value = HeaderValue::from_str(&session_id)
                .map_err(|error| Error::Protocol(format!("invalid MCP session ID: {error}")))?;
            headers.insert(HeaderName::from_static("mcp-session-id"), value);
        }
        if let Some(version) = self
            .inner
            .negotiated_protocol
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
        {
            let value = HeaderValue::from_str(&version).map_err(|error| {
                Error::Protocol(format!("invalid MCP protocol version: {error}"))
            })?;
            headers.insert(HeaderName::from_static("mcp-protocol-version"), value);
        }
        self.inner
            .client
            .post(self.inner.url.clone())
            .headers(headers)
            .json(message)
            .send()
            .await
            .map_err(Error::Http)
    }

    fn base_headers(&self) -> HeaderMap {
        self.inner.headers.clone()
    }

    fn capture_session(&self, headers: &HeaderMap) -> Result<()> {
        if let Some(value) = headers.get("mcp-session-id") {
            let value = value.to_str().map_err(|error| {
                Error::Protocol(format!("invalid MCP session ID header: {error}"))
            })?;
            if !value.is_empty() {
                *self
                    .inner
                    .session_id
                    .write()
                    .unwrap_or_else(|error| error.into_inner()) = Some(value.to_owned());
            }
        }
        Ok(())
    }

    fn capture_protocol(&self, headers: &HeaderMap) -> Result<()> {
        if let Some(value) = headers.get("mcp-protocol-version") {
            let value = value.to_str().map_err(|error| {
                Error::Protocol(format!("invalid MCP protocol version header: {error}"))
            })?;
            if !value.is_empty() {
                *self
                    .inner
                    .negotiated_protocol
                    .write()
                    .unwrap_or_else(|error| error.into_inner()) = Some(value.to_owned());
            }
        }
        Ok(())
    }

    async fn read_sse_response(
        &self,
        mut response: Response,
        id: u64,
        method: &str,
    ) -> Result<Value> {
        let mut buffer = Vec::new();
        loop {
            let Some(chunk) = response.chunk().await.map_err(Error::Http)? else {
                break;
            };
            buffer.extend_from_slice(&chunk);
            for data in drain_sse_data(&mut buffer)? {
                let message: Value = serde_json::from_str(&data)?;
                if let Some(response) = extract_response(&message, id)? {
                    return Ok(response);
                }
            }
        }
        if !buffer.is_empty() {
            buffer.push(b'\n');
            for data in drain_sse_data(&mut buffer)? {
                let message: Value = serde_json::from_str(&data)?;
                if let Some(response) = extract_response(&message, id)? {
                    return Ok(response);
                }
            }
        }
        Err(Error::Protocol(format!(
            "MCP HTTP SSE stream ended before response for `{method}`"
        )))
    }
}

fn extract_response(value: &Value, id: u64) -> Result<Option<Value>> {
    if let Some(messages) = value.as_array() {
        for message in messages {
            if let Some(response) = extract_response(message, id)? {
                return Ok(Some(response));
            }
        }
        return Ok(None);
    }
    if value.get("id") != Some(&json!(id)) {
        return Ok(None);
    }
    if let Some(error) = value.get("error") {
        let code = error
            .get("code")
            .and_then(Value::as_i64)
            .unwrap_or_default();
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("unknown MCP error");
        return Err(Error::Protocol(format!("JSON-RPC error {code}: {message}")));
    }
    let result = value.get("result").cloned().ok_or_else(|| {
        Error::Protocol("MCP response contains neither result nor error".to_owned())
    })?;
    Ok(Some(result))
}

fn drain_sse_data(buffer: &mut Vec<u8>) -> Result<Vec<String>> {
    let mut payloads = Vec::new();
    let mut consumed = 0usize;
    while let Some(relative) = buffer[consumed..].iter().position(|byte| *byte == b'\n') {
        let end = consumed + relative;
        let mut line = &buffer[consumed..end];
        if line.ends_with(b"\r") {
            line = &line[..line.len() - 1];
        }
        if let Some(payload) = line.strip_prefix(b"data:") {
            let payload = payload.strip_prefix(b" ").unwrap_or(payload);
            if !payload.is_empty() {
                let payload = std::str::from_utf8(payload).map_err(|error| {
                    Error::Protocol(format!("invalid UTF-8 in MCP SSE stream: {error}"))
                })?;
                payloads.push(payload.to_owned());
            }
        }
        consumed = end + 1;
    }
    buffer.drain(..consumed);
    Ok(payloads)
}

fn http_status_error(status: u16, body: &str) -> Error {
    Error::Protocol(format!(
        "MCP HTTP endpoint returned status {status}: {}",
        body.chars().take(2_000).collect::<String>()
    ))
}

/// One tool exposed by an HTTP MCP server.
pub struct McpHttpTool {
    client: McpHttpClient,
    remote_name: String,
    spec: ToolSpec,
}

impl std::fmt::Debug for McpHttpTool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("McpHttpTool")
            .field("spec", &self.spec)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Tool for McpHttpTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, arguments: Value, _context: &ToolContext) -> Result<ToolOutput> {
        self.client.call_tool(&self.remote_name, arguments).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_json_rpc_response_from_batch() {
        let value = json!([
            { "jsonrpc": "2.0", "method": "notifications/progress" },
            { "jsonrpc": "2.0", "id": 3, "result": { "ok": true } }
        ]);
        let result = extract_response(&value, 3).unwrap().unwrap();
        assert_eq!(result["ok"], true);
    }

    #[test]
    fn parses_sse_data_lines() {
        let mut buffer = b"event: message\ndata: {\"jsonrpc\":\"2.0\"}\n\n".to_vec();
        let payloads = drain_sse_data(&mut buffer).unwrap();
        assert_eq!(payloads, vec!["{\"jsonrpc\":\"2.0\"}"]);
    }
}
