use std::{collections::BTreeMap, time::Duration};

use async_trait::async_trait;
use reqwest::{
    Client,
    header::{AUTHORIZATION, HeaderMap, HeaderValue},
};
use serde_json::{Map, Value, json};

use crate::{
    ContentPart, Error, FinishReason, Message, Model, ModelDelta, ModelRequest, ModelResponse,
    Result, Role, ToolSpec, Usage,
    provider::{
        build_http_client, decode_tool_arguments, ensure_success, insert_extra_headers,
        json_response, read_sse, resolve_endpoint, validate_provider_config,
    },
};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const MIN_OUTPUT_TOKENS: u32 = 16;

/// An OpenAI Responses API provider.
#[derive(Clone)]
pub struct OpenAiResponses {
    client: Client,
    model: String,
    endpoint: String,
    context_window: Option<usize>,
}

impl std::fmt::Debug for OpenAiResponses {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAiResponses")
            .field("model", &self.model)
            .field("endpoint", &self.endpoint)
            .field("context_window", &self.context_window)
            .finish_non_exhaustive()
    }
}

impl OpenAiResponses {
    /// Starts configuring a Responses API provider.
    pub fn builder(model: impl Into<String>) -> OpenAiResponsesBuilder {
        OpenAiResponsesBuilder::new(model)
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

    fn request_body(request: &ModelRequest, model: &str, stream: bool) -> Value {
        let mut body = Map::new();
        body.insert("model".to_owned(), Value::String(model.to_owned()));
        body.insert("stream".to_owned(), Value::Bool(stream));
        body.insert("store".to_owned(), Value::Bool(false));

        if let Some(instructions) = collect_instructions(&request.messages) {
            body.insert("instructions".to_owned(), Value::String(instructions));
        }
        body.insert(
            "input".to_owned(),
            Value::Array(wire_input(&request.messages)),
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
        if let Some(max_tokens) = request.max_tokens {
            body.insert(
                "max_output_tokens".to_owned(),
                json!(max_tokens.max(MIN_OUTPUT_TOKENS)),
            );
        }
        Value::Object(body)
    }
}

#[async_trait]
impl Model for OpenAiResponses {
    fn name(&self) -> &str {
        &self.model
    }

    fn context_window(&self) -> Option<usize> {
        self.context_window
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse> {
        let body = Self::request_body(&request, &self.model, false);
        let response = self.client.post(&self.endpoint).json(&body).send().await?;
        let raw = json_response(response).await?;
        parse_response(&raw)
    }

    async fn complete_stream(
        &self,
        request: ModelRequest,
        on_delta: &(dyn Fn(ModelDelta) + Send + Sync),
    ) -> Result<ModelResponse> {
        let body = Self::request_body(&request, &self.model, true);
        let response = self.client.post(&self.endpoint).json(&body).send().await?;
        let response = ensure_success(response).await?;

        let mut stream = ResponsesStream::default();
        read_sse(response, |data| stream.consume(data, on_delta)).await?;
        stream.finish(&self.model)
    }
}

/// Builder for [`OpenAiResponses`].
pub struct OpenAiResponsesBuilder {
    model: String,
    api_key: Option<String>,
    base_url: String,
    context_window: Option<usize>,
    timeout: Duration,
    headers: BTreeMap<String, String>,
}

impl OpenAiResponsesBuilder {
    fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: None,
            base_url: DEFAULT_BASE_URL.to_owned(),
            context_window: None,
            timeout: Duration::from_secs(120),
            headers: BTreeMap::new(),
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

    /// Builds the provider.
    pub fn build(self) -> Result<OpenAiResponses> {
        validate_provider_config(&self.model, &self.base_url)?;

        let mut headers = HeaderMap::new();
        if let Some(api_key) = self.api_key.filter(|key| !key.trim().is_empty()) {
            let value = HeaderValue::from_str(&format!("Bearer {}", api_key.trim()))
                .map_err(|error| Error::Configuration(format!("invalid API key: {error}")))?;
            headers.insert(AUTHORIZATION, value);
        }
        insert_extra_headers(&mut headers, self.headers)?;

        let client = build_http_client(headers, self.timeout)?;
        let endpoint = resolve_endpoint(&self.base_url, "/responses", &["/responses"]);
        Ok(OpenAiResponses {
            client,
            model: self.model,
            endpoint,
            context_window: self.context_window,
        })
    }
}

impl std::fmt::Debug for OpenAiResponsesBuilder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAiResponsesBuilder")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("context_window", &self.context_window)
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

fn collect_instructions(messages: &[Message]) -> Option<String> {
    let instructions = messages
        .iter()
        .filter(|message| message.role == Role::System)
        .map(Message::text)
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    (!instructions.is_empty()).then_some(instructions)
}

fn wire_input(messages: &[Message]) -> Vec<Value> {
    let mut items = Vec::new();
    for message in messages {
        match message.role {
            Role::System => {}
            Role::Tool => {
                if let Some(call_id) = &message.tool_call_id {
                    items.push(json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": message.text()
                    }));
                }
            }
            Role::User => {
                items.push(json!({
                    "role": "user",
                    "content": wire_user_content(&message.content)
                }));
            }
            Role::Assistant => {
                let text = message.text();
                if !text.is_empty() {
                    items.push(json!({ "role": "assistant", "content": text }));
                }
                for call in &message.tool_calls {
                    items.push(json!({
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": call.arguments.to_string(),
                        "status": "completed"
                    }));
                }
            }
        }
    }
    items
}

fn wire_user_content(content: &[ContentPart]) -> Value {
    if let [ContentPart::Text { text }] = content {
        return Value::String(text.clone());
    }
    Value::Array(
        content
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => json!({ "type": "input_text", "text": text }),
                ContentPart::Image { url, .. } => {
                    json!({ "type": "input_image", "image_url": url })
                }
            })
            .collect(),
    )
}

fn wire_tool(tool: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
        "strict": false
    })
}

fn wire_tool_choice(choice: &crate::ToolChoice) -> Value {
    match choice {
        crate::ToolChoice::Auto => Value::String("auto".to_owned()),
        crate::ToolChoice::None => Value::String("none".to_owned()),
        crate::ToolChoice::Required => Value::String("required".to_owned()),
        crate::ToolChoice::Named(name) => json!({
            "type": "function",
            "name": name
        }),
    }
}

fn parse_response(raw: &Value) -> Result<ModelResponse> {
    if raw.get("status").and_then(Value::as_str) == Some("failed") {
        let message = raw
            .pointer("/error/message")
            .and_then(Value::as_str)
            .unwrap_or("OpenAI Responses request failed");
        return Err(Error::Model(message.to_owned()));
    }
    if let Some(message) = raw.pointer("/error/message").and_then(Value::as_str) {
        return Err(Error::Model(message.to_owned()));
    }
    let output = raw
        .get("output")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for item in output {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                for content in item
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if matches!(
                        content.get("type").and_then(Value::as_str),
                        Some("output_text" | "text")
                    ) && let Some(part) = content.get("text").and_then(Value::as_str)
                    {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(part);
                    }
                }
            }
            Some("function_call") => {
                let arguments = decode_tool_arguments(
                    item.get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                );
                tool_calls.push(crate::ToolCall {
                    id: item
                        .get("call_id")
                        .or_else(|| item.get("id"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    name: item
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    arguments,
                });
            }
            _ => {}
        }
    }
    Ok(ModelResponse {
        message: Message {
            role: Role::Assistant,
            content: (!text.is_empty())
                .then_some(vec![ContentPart::text(text)])
                .unwrap_or_default(),
            tool_calls,
            tool_call_id: None,
            name: None,
        },
        finish_reason: finish_reason(raw),
        usage: parse_usage(raw.get("usage")),
        model: raw
            .get("model")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        raw: Some(raw.clone()),
    })
}

fn finish_reason(raw: &Value) -> FinishReason {
    if raw.get("status").and_then(Value::as_str) == Some("incomplete") {
        return match raw
            .pointer("/incomplete_details/reason")
            .and_then(Value::as_str)
        {
            Some("max_output_tokens") => FinishReason::Length,
            Some("content_filter") => FinishReason::ContentFilter,
            Some(reason) => FinishReason::Other(reason.to_owned()),
            None => FinishReason::Length,
        };
    }
    if raw
        .get("output")
        .and_then(Value::as_array)
        .is_some_and(|items| {
            items
                .iter()
                .any(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
        })
    {
        FinishReason::ToolCalls
    } else {
        FinishReason::Stop
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
struct ResponsesStream {
    text: String,
    tool_calls: BTreeMap<usize, AccumulatedToolCall>,
    final_response: Option<Value>,
    terminal: bool,
}

impl ResponsesStream {
    fn consume(
        &mut self,
        data: &str,
        on_delta: &(dyn Fn(ModelDelta) + Send + Sync),
    ) -> Result<bool> {
        let value: Value = serde_json::from_str(data)?;
        match value.get("type").and_then(Value::as_str) {
            Some("response.output_text.delta") => {
                if let Some(text) = value.get("delta").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    self.text.push_str(text);
                    on_delta(ModelDelta::Text {
                        text: text.to_owned(),
                    });
                }
            }
            Some("response.refusal.delta") => {
                if let Some(text) = value.get("delta").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    self.text.push_str(text);
                    on_delta(ModelDelta::Text {
                        text: text.to_owned(),
                    });
                }
            }
            Some("response.output_item.added") => {
                self.consume_item_added(&value, on_delta)?;
            }
            Some("response.function_call_arguments.delta") => {
                let index = value
                    .get("output_index")
                    .and_then(Value::as_u64)
                    .unwrap_or_default() as usize;
                let arguments = value
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let call = self.tool_calls.entry(index).or_default();
                call.arguments.push_str(arguments);
                if !arguments.is_empty() {
                    on_delta(ModelDelta::ToolCall {
                        index,
                        id: None,
                        name: None,
                        arguments: Some(arguments.to_owned()),
                    });
                }
            }
            Some("response.function_call_arguments.done") => {
                let index = value
                    .get("output_index")
                    .and_then(Value::as_u64)
                    .unwrap_or_default() as usize;
                let arguments = value
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let call = self.tool_calls.entry(index).or_default();
                if !arguments.is_empty() && arguments != call.arguments {
                    let suffix = arguments.strip_prefix(&call.arguments);
                    call.arguments = arguments.to_owned();
                    if let Some(suffix) = suffix.filter(|suffix| !suffix.is_empty()) {
                        on_delta(ModelDelta::ToolCall {
                            index,
                            id: None,
                            name: None,
                            arguments: Some(suffix.to_owned()),
                        });
                    }
                }
            }
            Some("response.output_item.done") => {
                if let Some(item) = value.get("item")
                    && item.get("type").and_then(Value::as_str) == Some("function_call")
                {
                    let index = value
                        .get("output_index")
                        .and_then(Value::as_u64)
                        .unwrap_or_default() as usize;
                    let call = self.tool_calls.entry(index).or_default();
                    if call.id.is_empty() {
                        call.id = item
                            .get("call_id")
                            .or_else(|| item.get("id"))
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned();
                    }
                    if call.name.is_empty() {
                        call.name = item
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned();
                    }
                    if let Some(arguments) = item.get("arguments").and_then(Value::as_str)
                        && !arguments.is_empty()
                    {
                        call.arguments = arguments.to_owned();
                    }
                } else if self.text.is_empty()
                    && let Some(item) = value.get("item")
                    && item.get("type").and_then(Value::as_str) == Some("message")
                {
                    let text = item
                        .get("content")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .filter(|part| {
                            matches!(
                                part.get("type").and_then(Value::as_str),
                                Some("output_text" | "refusal" | "text")
                            )
                        })
                        .filter_map(|part| part.get("text").and_then(Value::as_str))
                        .collect::<String>();
                    if !text.is_empty() {
                        self.text = text.clone();
                        on_delta(ModelDelta::Text { text });
                    }
                }
            }
            Some("response.completed" | "response.incomplete") => {
                self.terminal = true;
                if let Some(response) = value.get("response") {
                    self.final_response = Some(response.clone());
                }
                return Ok(true);
            }
            Some("response.failed" | "response.error" | "error") => {
                let message = value
                    .pointer("/response/error/message")
                    .and_then(Value::as_str)
                    .or_else(|| value.pointer("/error/message").and_then(Value::as_str))
                    .or_else(|| value.get("message").and_then(Value::as_str))
                    .unwrap_or("OpenAI Responses stream failed");
                return Err(Error::Model(message.to_owned()));
            }
            _ => {}
        }
        Ok(false)
    }

    fn consume_item_added(
        &mut self,
        value: &Value,
        on_delta: &(dyn Fn(ModelDelta) + Send + Sync),
    ) -> Result<()> {
        let Some(item) = value.get("item") else {
            return Ok(());
        };
        if item.get("type").and_then(Value::as_str) != Some("function_call") {
            return Ok(());
        }
        let index = value
            .get("output_index")
            .and_then(Value::as_u64)
            .unwrap_or_default() as usize;
        let call = self.tool_calls.entry(index).or_default();
        call.id = item
            .get("call_id")
            .or_else(|| item.get("id"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        call.name = item
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        on_delta(ModelDelta::ToolCall {
            index,
            id: (!call.id.is_empty()).then(|| call.id.clone()),
            name: (!call.name.is_empty()).then(|| call.name.clone()),
            arguments: None,
        });
        Ok(())
    }

    fn finish(self, fallback_model: &str) -> Result<ModelResponse> {
        if !self.terminal {
            return Err(Error::InvalidModelResponse(
                "OpenAI Responses stream ended before a terminal event".to_owned(),
            ));
        }
        let response = self.final_response.ok_or_else(|| {
            Error::InvalidModelResponse(
                "OpenAI Responses terminal event contained no response".to_owned(),
            )
        })?;
        let mut parsed = parse_response(&response)?;

        if parsed.message.content.is_empty() && !self.text.is_empty() {
            parsed.message.content = vec![ContentPart::text(self.text)];
        }
        if parsed.message.tool_calls.is_empty() && !self.tool_calls.is_empty() {
            parsed.message.tool_calls = assembled_tool_calls(self.tool_calls);
        }
        if !parsed.message.tool_calls.is_empty() && parsed.finish_reason == FinishReason::Stop {
            parsed.finish_reason = FinishReason::ToolCalls;
        }
        if parsed.model.is_none() {
            parsed.model = Some(fallback_model.to_owned());
        }
        Ok(parsed)
    }
}

fn assembled_tool_calls(tool_calls: BTreeMap<usize, AccumulatedToolCall>) -> Vec<crate::ToolCall> {
    tool_calls
        .into_values()
        .map(|call| crate::ToolCall {
            id: call.id,
            name: call.name,
            arguments: decode_tool_arguments(&call.arguments),
        })
        .collect()
}

#[derive(Default)]
struct AccumulatedToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn encodes_responses_input_and_tools() {
        let request = ModelRequest {
            messages: vec![
                Message::system("Be concise."),
                Message::user("hello"),
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
            max_tokens: Some(123),
        };
        let body = OpenAiResponses::request_body(&request, "test-model", false);
        assert_eq!(body["instructions"], "Be concise.");
        assert_eq!(body["store"], false);
        assert_eq!(body["input"][1]["type"], "function_call");
        assert_eq!(body["input"][1]["status"], "completed");
        assert_eq!(body["input"][2]["type"], "function_call_output");
        assert_eq!(body["tools"][0]["name"], "echo");
        assert_eq!(body["max_output_tokens"], 123);
    }

    #[test]
    fn clamps_unsupported_small_output_budgets() {
        let mut request = ModelRequest {
            messages: vec![Message::user("hello")],
            tools: Vec::new(),
            tool_choice: crate::ToolChoice::Auto,
            temperature: None,
            max_tokens: Some(4),
        };
        let body = OpenAiResponses::request_body(&request, "test-model", false);
        assert_eq!(body["max_output_tokens"], 16);

        request.max_tokens = None;
        let body = OpenAiResponses::request_body(&request, "test-model", false);
        assert!(body.get("max_output_tokens").is_none());
    }

    #[test]
    fn decodes_responses_output() {
        let raw = json!({
            "id": "resp-1",
            "model": "test-model",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "hello"}]
                },
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "echo",
                    "arguments": "{\"text\":\"hi\"}"
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 3}
        });
        let response = parse_response(&raw).unwrap();
        assert_eq!(response.message.text(), "hello");
        assert_eq!(response.message.tool_calls[0].name, "echo");
        assert_eq!(response.finish_reason, FinishReason::ToolCalls);
        assert_eq!(response.usage.input_tokens, 10);
    }

    #[test]
    fn assembles_responses_stream() {
        let deltas = Mutex::new(Vec::new());
        let on_delta = |delta| deltas.lock().unwrap().push(delta);
        let mut stream = ResponsesStream::default();

        stream
            .consume(
                r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call-1","name":"echo"}}"#,
                &on_delta,
            )
            .unwrap();
        stream
            .consume(
                r#"{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"text\":"}"#,
                &on_delta,
            )
            .unwrap();
        stream
            .consume(
                r#"{"type":"response.function_call_arguments.delta","output_index":0,"delta":"\"hi\"}"}"#,
                &on_delta,
            )
            .unwrap();
        stream
            .consume(
                r#"{"type":"response.completed","response":{"model":"test","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"done"}]},{"type":"function_call","call_id":"call-1","name":"echo","arguments":"{\"text\":\"hi\"}"}],"usage":{"input_tokens":8,"output_tokens":2}}}"#,
                &on_delta,
            )
            .unwrap();

        let response = stream.finish("fallback").unwrap();
        assert_eq!(response.message.text(), "done");
        assert_eq!(
            response.message.tool_calls[0].arguments,
            json!({ "text": "hi" })
        );
        assert_eq!(response.finish_reason, FinishReason::ToolCalls);
        assert_eq!(response.usage.input_tokens, 8);
        assert!(matches!(
            deltas.lock().unwrap().as_slice(),
            [
                ModelDelta::ToolCall { .. },
                ModelDelta::ToolCall {
                    arguments: Some(_),
                    ..
                },
                ModelDelta::ToolCall {
                    arguments: Some(_),
                    ..
                }
            ]
        ));
    }

    #[test]
    fn rejects_stream_without_terminal_event() {
        let on_delta = |_delta| {};
        let mut stream = ResponsesStream::default();
        stream
            .consume(
                r#"{"type":"response.output_text.delta","delta":"partial"}"#,
                &on_delta,
            )
            .unwrap();

        let error = stream.finish("test").unwrap_err();
        assert!(error.to_string().contains("terminal event"));
    }

    #[test]
    fn falls_back_to_accumulated_tool_call_when_terminal_output_is_empty() {
        let on_delta = |_delta| {};
        let mut stream = ResponsesStream::default();
        for data in [
            r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call-1","name":"echo"}}"#,
            r#"{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"text\":\"hi\"}"}"#,
            r#"{"type":"response.completed","response":{"status":"completed","output":[],"usage":{"input_tokens":4,"output_tokens":1}}}"#,
        ] {
            stream.consume(data, &on_delta).unwrap();
        }

        let response = stream.finish("test-model").unwrap();
        assert_eq!(response.message.tool_calls[0].name, "echo");
        assert_eq!(
            response.message.tool_calls[0].arguments,
            json!({"text": "hi"})
        );
        assert_eq!(response.finish_reason, FinishReason::ToolCalls);
    }
}
