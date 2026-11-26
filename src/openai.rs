use std::{collections::BTreeMap, time::Duration};

use async_trait::async_trait;
use reqwest::{
    Client,
    header::{AUTHORIZATION, HeaderMap, HeaderName, HeaderValue},
};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::{
    ContentPart, Error, FinishReason, Message, Model, ModelDelta, ModelRequest, ModelResponse,
    Result, Role, ToolSpec, Usage,
    provider::{ensure_success, json_response, read_sse},
};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// An OpenAI-compatible Chat Completions provider.
///
/// This also works with local and third-party servers that implement the same
/// wire format. Vendor differences belong in another [`Model`] implementation,
/// not in the agent core.
#[derive(Clone)]
pub struct OpenAiCompatible {
    client: Client,
    model: String,
    endpoint: String,
    context_window: Option<usize>,
    include_stream_usage: bool,
}

impl std::fmt::Debug for OpenAiCompatible {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAiCompatible")
            .field("model", &self.model)
            .field("endpoint", &self.endpoint)
            .field("context_window", &self.context_window)
            .field("include_stream_usage", &self.include_stream_usage)
            .finish_non_exhaustive()
    }
}

impl OpenAiCompatible {
    /// Starts configuring a provider for a model name.
    pub fn builder(model: impl Into<String>) -> OpenAiCompatibleBuilder {
        OpenAiCompatibleBuilder::new(model)
    }

    /// Reads `OPENAI_API_KEY` and `OPENAI_BASE_URL`.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let mut builder = Self::builder(model);
        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            builder = builder.api_key(api_key);
        }
        if let Ok(base_url) = std::env::var("OPENAI_BASE_URL") {
            builder = builder.base_url(base_url);
        }
        builder.build()
    }

    fn request_body(
        request: &ModelRequest,
        model: &str,
        stream: bool,
        include_stream_usage: bool,
    ) -> Value {
        let messages = request
            .messages
            .iter()
            .map(wire_message)
            .collect::<Vec<_>>();
        let mut body = Map::new();
        body.insert("model".to_owned(), Value::String(model.to_owned()));
        body.insert("messages".to_owned(), Value::Array(messages));
        body.insert("stream".to_owned(), Value::Bool(stream));
        if stream && include_stream_usage {
            body.insert(
                "stream_options".to_owned(),
                json!({ "include_usage": true }),
            );
        }

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
        if let Some(max_tokens) = request.max_tokens {
            body.insert("max_tokens".to_owned(), json!(max_tokens));
        }
        Value::Object(body)
    }
}

#[async_trait]
impl Model for OpenAiCompatible {
    fn name(&self) -> &str {
        &self.model
    }

    fn context_window(&self) -> Option<usize> {
        self.context_window
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse> {
        let body = Self::request_body(&request, &self.model, false, false);
        let response = self.client.post(&self.endpoint).json(&body).send().await?;
        let raw = json_response(response).await?;
        let parsed: ChatResponse = serde_json::from_value(raw.clone())?;
        let choice = parsed.choices.into_iter().next().ok_or_else(|| {
            Error::InvalidModelResponse("provider returned no choices".to_owned())
        })?;

        let tool_calls = choice
            .message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|call| crate::ToolCall {
                id: call.id,
                name: call.function.name,
                arguments: serde_json::from_str(&call.function.arguments)
                    .unwrap_or(Value::String(call.function.arguments)),
            })
            .collect::<Vec<_>>();
        let message = Message {
            role: Role::Assistant,
            content: choice
                .message
                .content
                .map(parse_content)
                .unwrap_or_default(),
            tool_calls,
            tool_call_id: None,
            name: None,
        };

        Ok(ModelResponse {
            message,
            finish_reason: parse_finish_reason(choice.finish_reason.as_deref()),
            usage: parsed.usage.map(Into::into).unwrap_or_default(),
            model: parsed.model,
            raw: Some(raw),
        })
    }

    async fn complete_stream(
        &self,
        request: ModelRequest,
        on_delta: &(dyn Fn(ModelDelta) + Send + Sync),
    ) -> Result<ModelResponse> {
        let body = Self::request_body(&request, &self.model, true, self.include_stream_usage);
        let response = self.client.post(&self.endpoint).json(&body).send().await?;
        let response = ensure_success(response).await?;

        let mut accumulator = StreamAccumulator::default();
        let mut saw_data = false;
        read_sse(response, |data| {
            if data.trim() == "[DONE]" {
                return Ok(true);
            }
            accumulator.consume(data, on_delta)?;
            saw_data = true;
            Ok(false)
        })
        .await?;
        if !saw_data {
            return Err(Error::InvalidModelResponse(
                "streaming response contained no SSE data events".to_owned(),
            ));
        }

        Ok(accumulator.finish(&self.model))
    }
}

/// Builder for [`OpenAiCompatible`].
pub struct OpenAiCompatibleBuilder {
    model: String,
    api_key: Option<String>,
    base_url: String,
    context_window: Option<usize>,
    timeout: Duration,
    headers: BTreeMap<String, String>,
    include_stream_usage: bool,
}

impl OpenAiCompatibleBuilder {
    fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: None,
            base_url: DEFAULT_BASE_URL.to_owned(),
            context_window: None,
            timeout: Duration::from_secs(120),
            headers: BTreeMap::new(),
            include_stream_usage: true,
        }
    }

    /// Sets the bearer token.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Sets the API root, including `/v1` when required by the server.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Sets a context window used by automatic compression.
    pub fn context_window(mut self, tokens: usize) -> Self {
        self.context_window = Some(tokens);
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

    /// Requests token usage in the final streaming chunk.
    pub fn stream_usage(mut self, enabled: bool) -> Self {
        self.include_stream_usage = enabled;
        self
    }

    /// Builds the provider.
    pub fn build(self) -> Result<OpenAiCompatible> {
        if self.model.trim().is_empty() {
            return Err(Error::Configuration(
                "model name cannot be empty".to_owned(),
            ));
        }
        if self.base_url.trim().is_empty() {
            return Err(Error::Configuration("base URL cannot be empty".to_owned()));
        }

        let mut headers = HeaderMap::new();
        if let Some(api_key) = self.api_key.filter(|key| !key.trim().is_empty()) {
            let value = HeaderValue::from_str(&format!("Bearer {}", api_key.trim()))
                .map_err(|error| Error::Configuration(format!("invalid API key: {error}")))?;
            headers.insert(AUTHORIZATION, value);
        }
        for (name, value) in self.headers {
            let name = HeaderName::from_bytes(name.as_bytes()).map_err(|error| {
                Error::Configuration(format!("invalid HTTP header name `{name}`: {error}"))
            })?;
            let value = HeaderValue::from_str(&value).map_err(|error| {
                Error::Configuration(format!("invalid HTTP header value: {error}"))
            })?;
            headers.insert(name, value);
        }

        let client = Client::builder()
            .default_headers(headers)
            .timeout(self.timeout)
            .build()?;
        let base = self.base_url.trim_end_matches('/');
        let endpoint = if base.ends_with("/chat/completions") {
            base.to_owned()
        } else {
            format!("{base}/chat/completions")
        };

        Ok(OpenAiCompatible {
            client,
            model: self.model,
            endpoint,
            context_window: self.context_window,
            include_stream_usage: self.include_stream_usage,
        })
    }
}

impl std::fmt::Debug for OpenAiCompatibleBuilder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAiCompatibleBuilder")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("context_window", &self.context_window)
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

fn wire_message(message: &Message) -> Value {
    let role = message.role.to_string();
    let mut object = Map::new();
    object.insert("role".to_owned(), Value::String(role.clone()));

    if message.role == Role::Tool {
        object.insert("content".to_owned(), Value::String(message.text()));
        if let Some(call_id) = &message.tool_call_id {
            object.insert("tool_call_id".to_owned(), Value::String(call_id.clone()));
        }
        return Value::Object(object);
    }

    object.insert("content".to_owned(), wire_content(&message.content));
    if !message.tool_calls.is_empty() {
        object.insert(
            "tool_calls".to_owned(),
            Value::Array(
                message
                    .tool_calls
                    .iter()
                    .map(|call| {
                        json!({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments.to_string()
                            }
                        })
                    })
                    .collect(),
            ),
        );
    }
    Value::Object(object)
}

fn wire_content(content: &[ContentPart]) -> Value {
    if content.is_empty() {
        return Value::Null;
    }
    if let [ContentPart::Text { text }] = content {
        return Value::String(text.clone());
    }
    Value::Array(
        content
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => json!({ "type": "text", "text": text }),
                ContentPart::Image { url, .. } => {
                    json!({ "type": "image_url", "image_url": { "url": url } })
                }
            })
            .collect(),
    )
}

fn wire_tool(tool: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema
        }
    })
}

fn wire_tool_choice(choice: &crate::ToolChoice) -> Value {
    match choice {
        crate::ToolChoice::Auto => Value::String("auto".to_owned()),
        crate::ToolChoice::None => Value::String("none".to_owned()),
        crate::ToolChoice::Required => Value::String("required".to_owned()),
        crate::ToolChoice::Named(name) => json!({
            "type": "function",
            "function": { "name": name }
        }),
    }
}

fn parse_content(value: Value) -> Vec<ContentPart> {
    match value {
        Value::String(text) => vec![ContentPart::text(text)],
        Value::Array(parts) => parts
            .into_iter()
            .filter_map(|part| {
                let text = part.get("text")?.as_str()?.to_owned();
                Some(ContentPart::text(text))
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn parse_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("stop") => FinishReason::Stop,
        Some("tool_calls") | Some("function_call") => FinishReason::ToolCalls,
        Some("length") => FinishReason::Length,
        Some("content_filter") => FinishReason::ContentFilter,
        Some(other) => FinishReason::Other(other.to_owned()),
        None => FinishReason::Other("missing".to_owned()),
    }
}

#[derive(Deserialize)]
struct ChatResponse {
    model: Option<String>,
    #[serde(default)]
    choices: Vec<Choice>,
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct Choice {
    message: WireAssistantMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct WireAssistantMessage {
    content: Option<Value>,
    #[serde(default)]
    tool_calls: Option<Vec<WireToolCall>>,
}

#[derive(Deserialize)]
struct WireToolCall {
    id: String,
    function: WireFunction,
}

#[derive(Deserialize)]
struct WireFunction {
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Deserialize)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

impl From<WireUsage> for Usage {
    fn from(value: WireUsage) -> Self {
        Self {
            input_tokens: value.prompt_tokens,
            output_tokens: value.completion_tokens,
        }
    }
}

#[derive(Default)]
struct StreamAccumulator {
    text: String,
    tool_calls: BTreeMap<usize, AccumulatedToolCall>,
    finish_reason: Option<String>,
    usage: Option<WireUsage>,
    model: Option<String>,
}

impl StreamAccumulator {
    fn consume(&mut self, data: &str, on_delta: &(dyn Fn(ModelDelta) + Send + Sync)) -> Result<()> {
        let value: Value = serde_json::from_str(data)?;
        if let Some(error) = value.get("error") {
            return Err(Error::Model(format!("stream error: {error}")));
        }
        let chunk: StreamChunk = serde_json::from_value(value)?;
        if let Some(model) = chunk.model {
            self.model = Some(model);
        }
        if let Some(usage) = chunk.usage {
            self.usage = Some(usage);
        }

        for choice in chunk.choices {
            if choice.index != 0 {
                continue;
            }
            if let Some(content) = choice.delta.content {
                let text = stream_text(content);
                if !text.is_empty() {
                    self.text.push_str(&text);
                    on_delta(ModelDelta::Text { text: text.clone() });
                }
            }
            for fragment in choice.delta.tool_calls.unwrap_or_default() {
                let accumulated = self.tool_calls.entry(fragment.index).or_default();
                let mut id = None;
                let mut name = None;
                let mut arguments = None;

                if let Some(fragment_id) = fragment.id.filter(|id| !id.is_empty()) {
                    accumulated.id = fragment_id.clone();
                    id = Some(fragment_id);
                }
                if let Some(function) = fragment.function {
                    if let Some(fragment_name) = function.name.filter(|name| !name.is_empty()) {
                        if accumulated.name.is_empty() {
                            accumulated.name = fragment_name.clone();
                        } else {
                            accumulated.name.push_str(&fragment_name);
                        }
                        name = Some(fragment_name);
                    }
                    if let Some(fragment_arguments) =
                        function.arguments.filter(|arguments| !arguments.is_empty())
                    {
                        accumulated.arguments.push_str(&fragment_arguments);
                        arguments = Some(fragment_arguments);
                    }
                }

                if id.is_some() || name.is_some() || arguments.is_some() {
                    on_delta(ModelDelta::ToolCall {
                        index: fragment.index,
                        id,
                        name,
                        arguments,
                    });
                }
            }
            if let Some(reason) = choice.finish_reason {
                self.finish_reason = Some(reason);
            }
        }
        Ok(())
    }

    fn finish(self, fallback_model: &str) -> ModelResponse {
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
        let content = if self.text.is_empty() {
            Vec::new()
        } else {
            vec![ContentPart::text(self.text)]
        };
        ModelResponse {
            message: Message {
                role: Role::Assistant,
                content,
                tool_calls,
                tool_call_id: None,
                name: None,
            },
            finish_reason: parse_finish_reason(self.finish_reason.as_deref()),
            usage: self.usage.map(Into::into).unwrap_or_default(),
            model: self.model.or_else(|| Some(fallback_model.to_owned())),
            raw: None,
        }
    }
}

#[derive(Default)]
struct AccumulatedToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct StreamChunk {
    model: Option<String>,
    #[serde(default)]
    choices: Vec<StreamChoice>,
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct StreamChoice {
    #[serde(default)]
    index: usize,
    delta: StreamMessageDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct StreamMessageDelta {
    content: Option<Value>,
    #[serde(default)]
    tool_calls: Option<Vec<StreamToolCall>>,
}

#[derive(Deserialize)]
struct StreamToolCall {
    #[serde(default)]
    index: usize,
    id: Option<String>,
    function: Option<StreamFunction>,
}

#[derive(Deserialize)]
struct StreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

fn stream_text(value: Value) -> String {
    match value {
        Value::String(text) => text,
        Value::Array(parts) => parts
            .into_iter()
            .filter_map(|part| {
                let text = part.get("text")?.as_str()?;
                Some(text.to_owned())
            })
            .collect(),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolChoice;
    use std::sync::Mutex;

    #[test]
    fn encodes_openai_messages_and_tools() {
        let message = Message::assistant_with_tool_calls(
            "",
            vec![crate::ToolCall {
                id: "call_1".to_owned(),
                name: "echo".to_owned(),
                arguments: json!({ "text": "hi" }),
            }],
        );
        let encoded = wire_message(&message);
        assert_eq!(encoded["role"], "assistant");
        assert_eq!(encoded["tool_calls"][0]["function"]["name"], "echo");
        assert_eq!(
            encoded["tool_calls"][0]["function"]["arguments"],
            "{\"text\":\"hi\"}"
        );

        let tool = ToolSpec::new(
            "echo",
            "Echo text",
            json!({ "type": "object", "properties": {} }),
        );
        let encoded = wire_tool(&tool);
        assert_eq!(encoded["type"], "function");
        assert_eq!(encoded["function"]["name"], "echo");
        assert!(matches!(
            wire_tool_choice(&ToolChoice::Auto),
            Value::String(value) if value == "auto"
        ));
    }

    #[test]
    fn decodes_text_and_tool_calls() {
        let raw: Value = json!({
            "model": "test",
            "choices": [{
                "message": {
                    "content": "hello",
                    "tool_calls": [{
                        "id": "1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": "{\"q\":\"rust\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 4 }
        });
        let parsed: ChatResponse = serde_json::from_value(raw).unwrap();
        let choice = parsed.choices.into_iter().next().unwrap();
        assert_eq!(
            choice.message.content.map(parse_content).unwrap()[0],
            ContentPart::text("hello")
        );
        let call = choice
            .message
            .tool_calls
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(call.function.name, "lookup");
        assert_eq!(
            parse_finish_reason(choice.finish_reason.as_deref()),
            FinishReason::ToolCalls
        );
    }

    #[test]
    fn assembles_streamed_text_tools_and_usage() {
        let deltas = Mutex::new(Vec::new());
        let on_delta = |delta| deltas.lock().unwrap().push(delta);
        let mut accumulator = StreamAccumulator::default();

        accumulator
            .consume(
                r#"{"model":"stream-model","choices":[{"index":0,"delta":{"content":"Hel"}}]}"#,
                &on_delta,
            )
            .unwrap();
        accumulator
            .consume(
                r#"{"choices":[{"index":0,"delta":{"content":"lo"}}]}"#,
                &on_delta,
            )
            .unwrap();
        accumulator
            .consume(
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"echo","arguments":"{\"te"}}]}}]}"#,
                &on_delta,
            )
            .unwrap();
        accumulator
            .consume(
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"xt\":\"hi\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":9,"completion_tokens":4}}"#,
                &on_delta,
            )
            .unwrap();

        let response = accumulator.finish("fallback");
        assert_eq!(response.message.text(), "Hello");
        assert_eq!(response.message.tool_calls[0].name, "echo");
        assert_eq!(
            response.message.tool_calls[0].arguments,
            json!({ "text": "hi" })
        );
        assert_eq!(response.usage.input_tokens, 9);
        assert_eq!(response.usage.output_tokens, 4);
        assert_eq!(response.finish_reason, FinishReason::ToolCalls);
        assert_eq!(response.model.as_deref(), Some("stream-model"));
        assert_eq!(deltas.lock().unwrap().len(), 4);
    }
}
