use std::{
    collections::{BTreeMap, BTreeSet},
    time::Duration,
};

use async_trait::async_trait;
use reqwest::{
    Client,
    header::{HeaderMap, HeaderName, HeaderValue},
};
use serde_json::{Map, Value, json};

use crate::{
    ContentPart, Error, FinishReason, Message, Model, ModelDelta, ModelRequest, ModelResponse,
    Result, Role, ToolSpec, Usage,
    provider::{
        build_http_client, ensure_success, insert_extra_headers, json_response,
        object_tool_arguments, parse_data_url, read_sse, resolve_versioned_endpoint,
        validate_provider_config,
    },
};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const DEFAULT_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4_096;

/// An Anthropic Messages API provider.
#[derive(Clone)]
pub struct Anthropic {
    client: Client,
    model: String,
    endpoint: String,
    context_window: Option<usize>,
    default_max_tokens: u32,
}

impl std::fmt::Debug for Anthropic {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Anthropic")
            .field("model", &self.model)
            .field("endpoint", &self.endpoint)
            .field("context_window", &self.context_window)
            .field("default_max_tokens", &self.default_max_tokens)
            .finish_non_exhaustive()
    }
}

impl Anthropic {
    /// Starts configuring an Anthropic provider.
    pub fn builder(model: impl Into<String>) -> AnthropicBuilder {
        AnthropicBuilder::new(model)
    }

    /// Reads `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, and `ANTHROPIC_VERSION`.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let mut builder = Self::builder(model);
        if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
            builder = builder.api_key(api_key);
        }
        if let Ok(base_url) = std::env::var("ANTHROPIC_BASE_URL") {
            builder = builder.base_url(base_url);
        }
        if let Ok(version) = std::env::var("ANTHROPIC_VERSION") {
            builder = builder.version(version);
        }
        builder.build()
    }

    fn request_body(&self, request: &ModelRequest, stream: bool) -> Value {
        let mut body = Map::new();
        body.insert("model".to_owned(), Value::String(self.model.clone()));
        body.insert("stream".to_owned(), Value::Bool(stream));
        body.insert(
            "max_tokens".to_owned(),
            json!(request.max_tokens.unwrap_or(self.default_max_tokens)),
        );
        if let Some(system) = collect_system(&request.messages) {
            body.insert("system".to_owned(), Value::String(system));
        }
        body.insert(
            "messages".to_owned(),
            Value::Array(wire_messages(&request.messages)),
        );
        if !request.tools.is_empty() {
            body.insert(
                "tools".to_owned(),
                Value::Array(request.tools.iter().map(wire_tool).collect()),
            );
            body.insert(
                "tool_choice".to_owned(),
                wire_tool_choice(&request.tool_choice),
            );
        }
        if let Some(temperature) = request.temperature {
            body.insert("temperature".to_owned(), json!(temperature));
        }
        Value::Object(body)
    }
}

#[async_trait]
impl Model for Anthropic {
    fn name(&self) -> &str {
        &self.model
    }

    fn context_window(&self) -> Option<usize> {
        self.context_window
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse> {
        let body = self.request_body(&request, false);
        let response = self.client.post(&self.endpoint).json(&body).send().await?;
        let raw = json_response(response).await?;
        parse_response(&raw)
    }

    async fn complete_stream(
        &self,
        request: ModelRequest,
        on_delta: &(dyn Fn(ModelDelta) + Send + Sync),
    ) -> Result<ModelResponse> {
        let body = self.request_body(&request, true);
        let response = self.client.post(&self.endpoint).json(&body).send().await?;
        let response = ensure_success(response).await?;

        let mut stream = AnthropicStream::default();
        read_sse(response, |data| stream.consume(data, on_delta)).await?;
        stream.finish(&self.model)
    }
}

/// Builder for [`Anthropic`].
pub struct AnthropicBuilder {
    model: String,
    api_key: Option<String>,
    base_url: String,
    version: String,
    context_window: Option<usize>,
    timeout: Duration,
    headers: BTreeMap<String, String>,
    default_max_tokens: u32,
}

impl AnthropicBuilder {
    fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: None,
            base_url: DEFAULT_BASE_URL.to_owned(),
            version: DEFAULT_VERSION.to_owned(),
            context_window: None,
            timeout: Duration::from_secs(120),
            headers: BTreeMap::new(),
            default_max_tokens: DEFAULT_MAX_TOKENS,
        }
    }

    /// Sets the Anthropic API key.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Sets the API root, including `/v1` when required by the server.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Sets the `anthropic-version` header.
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Sets a context window used by automatic compression.
    pub fn context_window(mut self, tokens: usize) -> Self {
        self.context_window = Some(tokens);
        self
    }

    /// Sets the default `max_tokens` when a request does not specify one.
    pub fn default_max_tokens(mut self, tokens: u32) -> Self {
        self.default_max_tokens = tokens.max(1);
        self
    }

    /// Sets the per-request timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Adds one request header.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    /// Builds the provider.
    pub fn build(self) -> Result<Anthropic> {
        validate_provider_config(&self.model, &self.base_url)?;

        let mut headers = HeaderMap::new();
        if let Some(api_key) = self.api_key.filter(|key| !key.trim().is_empty()) {
            let value = HeaderValue::from_str(api_key.trim())
                .map_err(|error| Error::Configuration(format!("invalid API key: {error}")))?;
            headers.insert(HeaderName::from_static("x-api-key"), value);
        }
        headers.insert(
            HeaderName::from_static("anthropic-version"),
            HeaderValue::from_str(&self.version).map_err(|error| {
                Error::Configuration(format!("invalid Anthropic version: {error}"))
            })?,
        );
        insert_extra_headers(&mut headers, self.headers)?;

        let client = build_http_client(headers, self.timeout)?;
        let endpoint =
            resolve_versioned_endpoint(&self.base_url, "/v1", "/messages", &["/messages"]);
        Ok(Anthropic {
            client,
            model: self.model,
            endpoint,
            context_window: self.context_window,
            default_max_tokens: self.default_max_tokens,
        })
    }
}

impl std::fmt::Debug for AnthropicBuilder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AnthropicBuilder")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("version", &self.version)
            .field("context_window", &self.context_window)
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

fn collect_system(messages: &[Message]) -> Option<String> {
    let system = messages
        .iter()
        .filter(|message| message.role == Role::System)
        .map(Message::text)
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    (!system.is_empty()).then_some(system)
}

fn wire_messages(messages: &[Message]) -> Vec<Value> {
    let mut output = Vec::new();
    let mut index = 0;
    while index < messages.len() {
        match messages[index].role {
            Role::System => index += 1,
            Role::Tool => {
                let mut blocks = Vec::new();
                while index < messages.len() && messages[index].role == Role::Tool {
                    if let Some(call_id) = &messages[index].tool_call_id {
                        let text = messages[index].text();
                        let (text, is_error) = match text.strip_prefix("Tool error: ") {
                            Some(error) => (error.to_owned(), true),
                            None => (text, false),
                        };
                        blocks.push(json!({
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": text,
                            "is_error": is_error
                        }));
                    }
                    index += 1;
                }
                if !blocks.is_empty() {
                    output.push(json!({ "role": "user", "content": blocks }));
                }
            }
            Role::User => {
                output.push(json!({
                    "role": "user",
                    "content": wire_content(&messages[index].content)
                }));
                index += 1;
            }
            Role::Assistant => {
                let message = &messages[index];
                let mut blocks = Vec::new();
                for part in &message.content {
                    if let ContentPart::Text { text } = part
                        && !text.is_empty()
                    {
                        blocks.push(json!({ "type": "text", "text": text }));
                    }
                }
                for call in &message.tool_calls {
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments
                    }));
                }
                if !blocks.is_empty() {
                    output.push(json!({ "role": "assistant", "content": blocks }));
                }
                index += 1;
            }
        }
    }
    output
}

fn wire_content(content: &[ContentPart]) -> Value {
    if let [ContentPart::Text { text }] = content {
        return Value::String(text.clone());
    }
    Value::Array(
        content
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => json!({ "type": "text", "text": text }),
                ContentPart::Image { url, mime_type } => match parse_data_url(url) {
                    Some((data_mime, data)) => json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type.as_deref().unwrap_or(&data_mime),
                            "data": data
                        }
                    }),
                    None => json!({
                        "type": "image",
                        "source": { "type": "url", "url": url }
                    }),
                },
            })
            .collect(),
    )
}

fn wire_tool(tool: &ToolSpec) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema
    })
}

fn wire_tool_choice(choice: &crate::ToolChoice) -> Value {
    match choice {
        crate::ToolChoice::Auto => json!({ "type": "auto" }),
        crate::ToolChoice::None => json!({ "type": "none" }),
        crate::ToolChoice::Required => json!({ "type": "any" }),
        crate::ToolChoice::Named(name) => json!({ "type": "tool", "name": name }),
    }
}

fn parse_response(raw: &Value) -> Result<ModelResponse> {
    let mut text = Vec::new();
    let mut tool_calls = Vec::new();
    for block in raw
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(value) = block.get("text").and_then(Value::as_str) {
                    text.push(value.to_owned());
                }
            }
            Some("tool_use") => {
                tool_calls.push(crate::ToolCall {
                    id: block
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    name: block
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    arguments: object_tool_arguments(
                        block.get("input").cloned().unwrap_or(Value::Null),
                    ),
                });
            }
            _ => {}
        }
    }
    Ok(ModelResponse {
        message: Message {
            role: Role::Assistant,
            content: (!text.is_empty())
                .then_some(vec![ContentPart::text(text.join("\n"))])
                .unwrap_or_default(),
            tool_calls,
            tool_call_id: None,
            name: None,
        },
        finish_reason: parse_finish_reason(raw.get("stop_reason").and_then(Value::as_str)),
        usage: parse_usage(raw.get("usage")),
        model: raw
            .get("model")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        raw: Some(raw.clone()),
    })
}

fn parse_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("end_turn" | "stop_sequence" | "pause_turn") => FinishReason::Stop,
        Some("max_tokens" | "model_context_window_exceeded") => FinishReason::Length,
        Some("tool_use") => FinishReason::ToolCalls,
        Some("refusal") => FinishReason::ContentFilter,
        Some(other) => FinishReason::Other(other.to_owned()),
        None => FinishReason::Other("missing".to_owned()),
    }
}

fn parse_usage(usage: Option<&Value>) -> Usage {
    let Some(usage) = usage else {
        return Usage::default();
    };
    Usage {
        input_tokens: usage
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        output_tokens: usage
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
    }
}

#[derive(Default)]
struct AnthropicStream {
    text: String,
    tool_calls: BTreeMap<usize, AccumulatedToolCall>,
    whole_tool_inputs: BTreeSet<usize>,
    stop_reason: Option<String>,
    input_tokens: u32,
    output_tokens: u32,
    model: Option<String>,
    stopped: bool,
}

impl AnthropicStream {
    fn consume(
        &mut self,
        data: &str,
        on_delta: &(dyn Fn(ModelDelta) + Send + Sync),
    ) -> Result<bool> {
        let value: Value = serde_json::from_str(data)?;
        match value.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                if let Some(message) = value.get("message") {
                    self.model = message
                        .get("model")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned);
                    self.input_tokens = message
                        .pointer("/usage/input_tokens")
                        .and_then(Value::as_u64)
                        .unwrap_or_default() as u32;
                    self.output_tokens = message
                        .pointer("/usage/output_tokens")
                        .and_then(Value::as_u64)
                        .unwrap_or_default() as u32;
                }
            }
            Some("content_block_start") => {
                let index = value
                    .get("index")
                    .and_then(Value::as_u64)
                    .unwrap_or_default() as usize;
                if let Some(block) = value.get("content_block") {
                    match block.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            if let Some(text) = block.get("text").and_then(Value::as_str)
                                && !text.is_empty()
                            {
                                self.text.push_str(text);
                                on_delta(ModelDelta::Text {
                                    text: text.to_owned(),
                                });
                            }
                        }
                        Some("tool_use") => {
                            let call = self.tool_calls.entry(index).or_default();
                            call.id = block
                                .get("id")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned();
                            call.name = block
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned();
                            let whole_input = block
                                .get("input")
                                .filter(|input| {
                                    input.as_object().is_some_and(|object| !object.is_empty())
                                })
                                .cloned();
                            let arguments = whole_input.map(|input| {
                                self.whole_tool_inputs.insert(index);
                                call.arguments = input.to_string();
                                call.arguments.clone()
                            });
                            on_delta(ModelDelta::ToolCall {
                                index,
                                id: (!call.id.is_empty()).then(|| call.id.clone()),
                                name: (!call.name.is_empty()).then(|| call.name.clone()),
                                arguments,
                            });
                        }
                        _ => {}
                    }
                }
            }
            Some("content_block_delta") => {
                let index = value
                    .get("index")
                    .and_then(Value::as_u64)
                    .unwrap_or_default() as usize;
                if let Some(delta) = value.get("delta") {
                    match delta.get("type").and_then(Value::as_str) {
                        Some("text_delta") => {
                            if let Some(text) = delta.get("text").and_then(Value::as_str)
                                && !text.is_empty()
                            {
                                self.text.push_str(text);
                                on_delta(ModelDelta::Text {
                                    text: text.to_owned(),
                                });
                            }
                        }
                        Some("input_json_delta") => {
                            if self.whole_tool_inputs.contains(&index) {
                                return Ok(false);
                            }
                            if let Some(fragment) =
                                delta.get("partial_json").and_then(Value::as_str)
                            {
                                let call = self.tool_calls.entry(index).or_default();
                                call.arguments.push_str(fragment);
                                if !fragment.is_empty() {
                                    on_delta(ModelDelta::ToolCall {
                                        index,
                                        id: None,
                                        name: None,
                                        arguments: Some(fragment.to_owned()),
                                    });
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            Some("message_delta") => {
                if let Some(reason) = value.pointer("/delta/stop_reason").and_then(Value::as_str) {
                    self.stop_reason = Some(reason.to_owned());
                }
                if let Some(tokens) = value.pointer("/usage/input_tokens").and_then(Value::as_u64) {
                    self.input_tokens = tokens as u32;
                }
                if let Some(tokens) = value
                    .pointer("/usage/output_tokens")
                    .and_then(Value::as_u64)
                {
                    self.output_tokens = tokens as u32;
                }
            }
            Some("message_stop") => {
                self.stopped = true;
                return Ok(true);
            }
            Some("error") => {
                let message = value
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .or_else(|| value.pointer("/error/type").and_then(Value::as_str))
                    .unwrap_or("Anthropic stream failed");
                return Err(Error::Model(message.to_owned()));
            }
            _ => {}
        }
        Ok(false)
    }

    fn finish(self, fallback_model: &str) -> Result<ModelResponse> {
        if !self.stopped {
            return Err(Error::InvalidModelResponse(
                "Anthropic stream ended before message_stop".to_owned(),
            ));
        }
        let tool_calls = self
            .tool_calls
            .into_values()
            .map(|call| crate::ToolCall {
                id: call.id,
                name: call.name,
                arguments: serde_json::from_str(&call.arguments)
                    .unwrap_or(Value::String(call.arguments)),
            })
            .collect::<Vec<_>>();
        Ok(ModelResponse {
            message: Message {
                role: Role::Assistant,
                content: (!self.text.is_empty())
                    .then_some(vec![ContentPart::text(self.text)])
                    .unwrap_or_default(),
                tool_calls,
                tool_call_id: None,
                name: None,
            },
            finish_reason: parse_finish_reason(self.stop_reason.as_deref()),
            usage: Usage {
                input_tokens: self.input_tokens,
                output_tokens: self.output_tokens,
            },
            model: Some(self.model.unwrap_or_else(|| fallback_model.to_owned())),
            raw: None,
        })
    }
}

#[derive(Default)]
struct AccumulatedToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    fn anthropic() -> Anthropic {
        Anthropic::builder("claude-test")
            .api_key("test")
            .build()
            .unwrap()
    }

    #[test]
    fn encodes_anthropic_messages_and_tool_results() {
        let request = ModelRequest {
            messages: vec![
                Message::system("Be concise."),
                Message::user("use tools"),
                Message::assistant_with_tool_calls(
                    "",
                    vec![crate::ToolCall {
                        id: "call-1".to_owned(),
                        name: "echo".to_owned(),
                        arguments: json!({ "text": "hi" }),
                    }],
                ),
                Message::tool_result("call-1", "ok"),
            ],
            tools: vec![ToolSpec::new(
                "echo",
                "Echo",
                json!({ "type": "object", "properties": {} }),
            )],
            tool_choice: crate::ToolChoice::Auto,
            temperature: Some(0.2),
            max_tokens: Some(100),
        };
        let body = anthropic().request_body(&request, false);
        assert_eq!(body["system"], "Be concise.");
        assert_eq!(body["max_tokens"], 100);
        assert_eq!(body["tools"][0]["name"], "echo");
        assert_eq!(body["messages"][1]["content"][0]["type"], "tool_use");
        assert_eq!(body["messages"][2]["content"][0]["type"], "tool_result");
    }

    #[test]
    fn normalizes_anthropic_base_url_and_data_url_images() {
        let provider = Anthropic::builder("claude-test")
            .base_url("https://api.anthropic.com")
            .build()
            .unwrap();
        assert_eq!(provider.endpoint, "https://api.anthropic.com/v1/messages");

        let request = ModelRequest {
            messages: vec![Message::user_with_content(vec![
                ContentPart::text("describe"),
                ContentPart::image("data:image/png;base64,aGVsbG8="),
            ])],
            tools: Vec::new(),
            tool_choice: crate::ToolChoice::Auto,
            temperature: None,
            max_tokens: None,
        };
        let body = provider.request_body(&request, false);
        assert_eq!(
            body["messages"][0]["content"][1]["source"]["type"],
            "base64"
        );
        assert_eq!(
            body["messages"][0]["content"][1]["source"]["media_type"],
            "image/png"
        );
        assert_eq!(
            body["messages"][0]["content"][1]["source"]["data"],
            "aGVsbG8="
        );
    }

    #[test]
    fn decodes_anthropic_response() {
        let raw = json!({
            "id": "msg-1",
            "model": "claude-test",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "tool_use", "id": "call-1", "name": "echo", "input": {"text": "hi"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 4}
        });
        let response = parse_response(&raw).unwrap();
        assert_eq!(response.message.text(), "hello");
        assert_eq!(response.message.tool_calls[0].name, "echo");
        assert_eq!(response.finish_reason, FinishReason::ToolCalls);
        assert_eq!(response.usage.input_tokens, 10);
    }

    #[test]
    fn assembles_anthropic_stream() {
        let deltas = Mutex::new(Vec::new());
        let on_delta = |delta| deltas.lock().unwrap().push(delta);
        let mut stream = AnthropicStream::default();
        for data in [
            r#"{"type":"message_start","message":{"model":"claude-test","usage":{"input_tokens":9}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hel"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}"#,
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call-1","name":"echo"}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"text\":\"hi\"}"}}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":4}}"#,
        ] {
            assert!(!stream.consume(data, &on_delta).unwrap());
        }
        assert!(
            stream
                .consume(r#"{"type":"message_stop"}"#, &on_delta)
                .unwrap()
        );
        let response = stream.finish("fallback").unwrap();
        assert_eq!(response.message.text(), "hello");
        assert_eq!(
            response.message.tool_calls[0].arguments,
            json!({ "text": "hi" })
        );
        assert_eq!(response.finish_reason, FinishReason::ToolCalls);
        assert_eq!(response.usage.output_tokens, 4);
    }

    #[test]
    fn accepts_tool_input_sent_whole_at_block_start() {
        let on_delta = |_delta| {};
        let mut stream = AnthropicStream::default();
        for data in [
            r#"{"type":"message_start","message":{"model":"claude-test","usage":{"input_tokens":3}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call-1","name":"echo","input":{"text":"hi"}}}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":2}}"#,
            r#"{"type":"message_stop"}"#,
        ] {
            stream.consume(data, &on_delta).unwrap();
        }

        let response = stream.finish("test").unwrap();
        assert_eq!(
            response.message.tool_calls[0].arguments,
            json!({"text": "hi"})
        );
    }

    #[test]
    fn rejects_stream_without_message_stop() {
        let on_delta = |_delta| {};
        let mut stream = AnthropicStream::default();
        stream
            .consume(
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}"#,
                &on_delta,
            )
            .unwrap();

        let error = stream.finish("test").unwrap_err();
        assert!(error.to_string().contains("message_stop"));
    }
}
