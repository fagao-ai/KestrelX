use std::{collections::BTreeMap, time::Duration};

use async_trait::async_trait;
use reqwest::{
    Client,
    header::{HeaderMap, HeaderName, HeaderValue},
};
use serde_json::{Map, Value, json};

use crate::{
    ContentPart, Error, FinishReason, Message, Model, ModelDelta, ModelRequest, ModelResponse,
    Result, Role, ToolSpec, Usage,
    provider::{ensure_success, json_response, read_sse},
};

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// A Google Gemini Generative Language API provider.
#[derive(Clone)]
pub struct Gemini {
    client: Client,
    model: String,
    base_url: String,
    context_window: Option<usize>,
}

impl std::fmt::Debug for Gemini {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Gemini")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("context_window", &self.context_window)
            .finish_non_exhaustive()
    }
}

impl Gemini {
    /// Starts configuring a Gemini provider.
    pub fn builder(model: impl Into<String>) -> GeminiBuilder {
        GeminiBuilder::new(model)
    }

    /// Reads `GEMINI_API_KEY` / `GOOGLE_API_KEY` and `GEMINI_BASE_URL`.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let mut builder = Self::builder(model);
        if let Ok(api_key) =
            std::env::var("GEMINI_API_KEY").or_else(|_| std::env::var("GOOGLE_API_KEY"))
        {
            builder = builder.api_key(api_key);
        }
        if let Ok(base_url) = std::env::var("GEMINI_BASE_URL") {
            builder = builder.base_url(base_url);
        }
        builder.build()
    }

    fn endpoint(&self, method: &str) -> String {
        let model = self.model.trim_start_matches('/');
        let model = if model.starts_with("models/") {
            model.to_owned()
        } else {
            format!("models/{model}")
        };
        format!("{}/{model}:{method}", self.base_url.trim_end_matches('/'))
    }

    fn request_body(&self, request: &ModelRequest) -> Value {
        let mut body = Map::new();
        body.insert(
            "contents".to_owned(),
            Value::Array(wire_contents(&request.messages)),
        );
        if let Some(system) = collect_system(&request.messages) {
            body.insert(
                "systemInstruction".to_owned(),
                json!({ "parts": [{ "text": system }] }),
            );
        }
        if !request.tools.is_empty() {
            body.insert(
                "tools".to_owned(),
                json!([{
                    "functionDeclarations": request
                        .tools
                        .iter()
                        .map(wire_tool)
                        .collect::<Vec<_>>()
                }]),
            );
            body.insert(
                "toolConfig".to_owned(),
                json!({ "functionCallingConfig": wire_tool_choice(&request.tool_choice) }),
            );
        }
        let mut generation = Map::new();
        if let Some(temperature) = request.temperature {
            generation.insert("temperature".to_owned(), json!(temperature));
        }
        if let Some(max_tokens) = request.max_tokens {
            generation.insert("maxOutputTokens".to_owned(), json!(max_tokens));
        }
        if !generation.is_empty() {
            body.insert("generationConfig".to_owned(), Value::Object(generation));
        }
        Value::Object(body)
    }
}

#[async_trait]
impl Model for Gemini {
    fn name(&self) -> &str {
        &self.model
    }

    fn context_window(&self) -> Option<usize> {
        self.context_window
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse> {
        let body = self.request_body(&request);
        let response = self
            .client
            .post(self.endpoint("generateContent"))
            .json(&body)
            .send()
            .await?;
        let raw = json_response(response).await?;
        parse_response(&raw)
    }

    async fn complete_stream(
        &self,
        request: ModelRequest,
        on_delta: &(dyn Fn(ModelDelta) + Send + Sync),
    ) -> Result<ModelResponse> {
        let body = self.request_body(&request);
        let response = self
            .client
            .post(format!(
                "{}?alt=sse",
                self.endpoint("streamGenerateContent")
            ))
            .json(&body)
            .send()
            .await?;
        let response = ensure_success(response).await?;
        let mut stream = GeminiStream::default();
        read_sse(response, |data| stream.consume(data, on_delta)).await?;
        stream.finish(&self.model)
    }
}

/// Builder for [`Gemini`].
pub struct GeminiBuilder {
    model: String,
    api_key: Option<String>,
    base_url: String,
    context_window: Option<usize>,
    timeout: Duration,
    headers: BTreeMap<String, String>,
}

impl GeminiBuilder {
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

    /// Sets the Google API key.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Sets the API base URL, normally ending in `/v1beta`.
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
    pub fn build(self) -> Result<Gemini> {
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
            let value = HeaderValue::from_str(api_key.trim())
                .map_err(|error| Error::Configuration(format!("invalid API key: {error}")))?;
            headers.insert(HeaderName::from_static("x-goog-api-key"), value);
        }
        for (name, value) in self.headers {
            let name = HeaderName::from_bytes(name.as_bytes()).map_err(|error| {
                Error::Configuration(format!("invalid HTTP header name `{name}`: {error}"))
            })?;
            let value = HeaderValue::from_str(&value)
                .map_err(|error| Error::Configuration(format!("invalid HTTP header: {error}")))?;
            headers.insert(name, value);
        }

        let client = Client::builder()
            .default_headers(headers)
            .timeout(self.timeout)
            .build()?;
        Ok(Gemini {
            client,
            model: self.model,
            base_url: self.base_url,
            context_window: self.context_window,
        })
    }
}

impl std::fmt::Debug for GeminiBuilder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GeminiBuilder")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
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

fn wire_contents(messages: &[Message]) -> Vec<Value> {
    let mut contents = Vec::new();
    let mut index = 0;
    while index < messages.len() {
        match messages[index].role {
            Role::System => index += 1,
            Role::User => {
                contents.push(json!({
                    "role": "user",
                    "parts": wire_parts(&messages[index].content)
                }));
                index += 1;
            }
            Role::Assistant => {
                let mut parts = wire_parts(&messages[index].content);
                parts.extend(messages[index].tool_calls.iter().enumerate().map(
                    |(call_index, call)| {
                        json!({
                            "functionCall": {
                                "id": if call.id.is_empty() {
                                    format!("call_{call_index}")
                                } else {
                                    call.id.clone()
                                },
                                "name": call.name,
                                "args": call.arguments
                            }
                        })
                    },
                ));
                if !parts.is_empty() {
                    contents.push(json!({ "role": "model", "parts": parts }));
                }
                index += 1;
            }
            Role::Tool => {
                let mut parts = Vec::new();
                while index < messages.len() && messages[index].role == Role::Tool {
                    parts.push(json!({
                        "functionResponse": {
                            "id": messages[index].tool_call_id,
                            "name": messages[index].name.clone().unwrap_or_else(|| "tool".to_owned()),
                            "response": { "output": messages[index].text() }
                        }
                    }));
                    index += 1;
                }
                if !parts.is_empty() {
                    contents.push(json!({ "role": "user", "parts": parts }));
                }
            }
        }
    }
    contents
}

fn wire_parts(content: &[ContentPart]) -> Vec<Value> {
    content
        .iter()
        .map(|part| match part {
            ContentPart::Text { text } => json!({ "text": text }),
            ContentPart::Image { url, mime_type } => json!({
                "fileData": {
                    "mimeType": mime_type.as_deref().unwrap_or("image/*"),
                    "fileUri": url
                }
            }),
        })
        .collect()
}

fn wire_tool(tool: &ToolSpec) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema
    })
}

fn wire_tool_choice(choice: &crate::ToolChoice) -> Value {
    match choice {
        crate::ToolChoice::Auto => json!({ "mode": "AUTO" }),
        crate::ToolChoice::None => json!({ "mode": "NONE" }),
        crate::ToolChoice::Required => json!({ "mode": "ANY" }),
        crate::ToolChoice::Named(name) => {
            json!({ "mode": "ANY", "allowedFunctionNames": [name] })
        }
    }
}

fn parse_response(raw: &Value) -> Result<ModelResponse> {
    let candidate = raw
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|candidates| candidates.first())
        .ok_or_else(|| Error::InvalidModelResponse("Gemini returned no candidates".to_owned()))?;
    let (text, tool_calls) = parse_parts(candidate);
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
        finish_reason: parse_finish_reason(candidate, raw),
        usage: parse_usage(raw.get("usageMetadata")),
        model: raw
            .get("modelVersion")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        raw: Some(raw.clone()),
    })
}

fn parse_parts(candidate: &Value) -> (String, Vec<crate::ToolCall>) {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for part in candidate
        .pointer("/content/parts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(value) = part.get("text").and_then(Value::as_str) {
            text.push_str(value);
        }
        if let Some(call) = part.get("functionCall") {
            let index = tool_calls.len();
            tool_calls.push(crate::ToolCall {
                id: call
                    .get("id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| format!("call_{index}")),
                name: call
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                arguments: call.get("args").cloned().unwrap_or(Value::Null),
            });
        }
    }
    (text, tool_calls)
}

fn parse_finish_reason(candidate: &Value, raw: &Value) -> FinishReason {
    if candidate
        .pointer("/content/parts")
        .and_then(Value::as_array)
        .is_some_and(|parts| parts.iter().any(|part| part.get("functionCall").is_some()))
    {
        return FinishReason::ToolCalls;
    }
    match candidate.get("finishReason").and_then(Value::as_str) {
        Some("STOP") => FinishReason::Stop,
        Some("MAX_TOKENS") => FinishReason::Length,
        Some(
            "SAFETY"
            | "RECITATION"
            | "BLOCKLIST"
            | "PROHIBITED_CONTENT"
            | "SPII"
            | "IMAGE_SAFETY"
            | "IMAGE_PROHIBITED_CONTENT",
        ) => FinishReason::ContentFilter,
        Some(reason @ ("MALFORMED_FUNCTION_CALL" | "UNEXPECTED_TOOL_CALL" | "OTHER")) => {
            FinishReason::Other(reason.to_lowercase())
        }
        Some(reason) => FinishReason::Other(reason.to_lowercase()),
        None if raw.get("promptFeedback").is_some() => FinishReason::ContentFilter,
        None => FinishReason::Other("missing".to_owned()),
    }
}

fn parse_usage(usage: Option<&Value>) -> Usage {
    let Some(usage) = usage else {
        return Usage::default();
    };
    Usage {
        input_tokens: usage
            .get("promptTokenCount")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
        output_tokens: usage
            .get("candidatesTokenCount")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32,
    }
}

#[derive(Default)]
struct GeminiStream {
    text: String,
    tool_calls: Vec<crate::ToolCall>,
    finish_reason: Option<FinishReason>,
    usage: Usage,
    model: Option<String>,
}

impl GeminiStream {
    fn consume(
        &mut self,
        data: &str,
        on_delta: &(dyn Fn(ModelDelta) + Send + Sync),
    ) -> Result<bool> {
        let value: Value = serde_json::from_str(data)?;
        if let Some(model) = value.get("modelVersion").and_then(Value::as_str) {
            self.model = Some(model.to_owned());
        }
        if let Some(usage) = value.get("usageMetadata") {
            self.usage = parse_usage(Some(usage));
        }
        if let Some(candidate) = value
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|candidates| candidates.first())
        {
            if let Some(parts) = candidate
                .pointer("/content/parts")
                .and_then(Value::as_array)
            {
                for part in parts {
                    if let Some(text) = part.get("text").and_then(Value::as_str)
                        && !text.is_empty()
                    {
                        self.text.push_str(text);
                        on_delta(ModelDelta::Text {
                            text: text.to_owned(),
                        });
                    }
                    if let Some(call) = part.get("functionCall") {
                        let index = self.tool_calls.len();
                        let call = crate::ToolCall {
                            id: call
                                .get("id")
                                .and_then(Value::as_str)
                                .map(ToOwned::to_owned)
                                .unwrap_or_else(|| format!("call_{index}")),
                            name: call
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned(),
                            arguments: call.get("args").cloned().unwrap_or(Value::Null),
                        };
                        on_delta(ModelDelta::ToolCall {
                            index,
                            id: Some(call.id.clone()),
                            name: Some(call.name.clone()),
                            arguments: Some(call.arguments.to_string()),
                        });
                        self.tool_calls.push(call);
                    }
                }
            }
            self.finish_reason = Some(parse_finish_reason(candidate, &value));
        }
        Ok(false)
    }

    fn finish(self, fallback_model: &str) -> Result<ModelResponse> {
        Ok(ModelResponse {
            message: Message {
                role: Role::Assistant,
                content: (!self.text.is_empty())
                    .then_some(vec![ContentPart::text(self.text)])
                    .unwrap_or_default(),
                tool_calls: self.tool_calls,
                tool_call_id: None,
                name: None,
            },
            finish_reason: self
                .finish_reason
                .unwrap_or(FinishReason::Other("missing".to_owned())),
            usage: self.usage,
            model: Some(self.model.unwrap_or_else(|| fallback_model.to_owned())),
            raw: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    fn gemini() -> Gemini {
        Gemini::builder("gemini-test")
            .api_key("test")
            .build()
            .unwrap()
    }

    #[test]
    fn encodes_gemini_tools_and_function_results() {
        let request = ModelRequest {
            messages: vec![
                Message::system("Be concise."),
                Message::user("use a tool"),
                Message::assistant_with_tool_calls(
                    "",
                    vec![crate::ToolCall {
                        id: "call-1".to_owned(),
                        name: "echo".to_owned(),
                        arguments: json!({ "text": "hi" }),
                    }],
                ),
                Message::tool_result("call-1", "ok").with_name("echo"),
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
        let body = gemini().request_body(&request);
        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "Be concise.");
        assert_eq!(
            body["contents"][1]["parts"][0]["functionCall"]["name"],
            "echo"
        );
        assert_eq!(
            body["contents"][2]["parts"][0]["functionResponse"]["name"],
            "echo"
        );
        assert_eq!(body["tools"][0]["functionDeclarations"][0]["name"], "echo");
        assert_eq!(body["generationConfig"]["maxOutputTokens"], 100);
    }

    #[test]
    fn decodes_gemini_response() {
        let raw = json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "hello"},
                        {"functionCall": {"id": "call-1", "name": "echo", "args": {"text": "hi"}}}
                    ]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4},
            "modelVersion": "gemini-test"
        });
        let response = parse_response(&raw).unwrap();
        assert_eq!(response.message.text(), "hello");
        assert_eq!(response.message.tool_calls[0].id, "call-1");
        assert_eq!(response.finish_reason, FinishReason::ToolCalls);
        assert_eq!(response.usage.output_tokens, 4);
    }

    #[test]
    fn assembles_gemini_stream() {
        let deltas = Mutex::new(Vec::new());
        let on_delta = |delta| deltas.lock().unwrap().push(delta);
        let mut stream = GeminiStream::default();
        stream
            .consume(
                r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"hel"}]}}],"usageMetadata":{"promptTokenCount":8}}"#,
                &on_delta,
            )
            .unwrap();
        stream
            .consume(
                r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"lo"},{"functionCall":{"id":"call-1","name":"echo","args":{"text":"hi"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":8,"candidatesTokenCount":4},"modelVersion":"gemini-test"}"#,
                &on_delta,
            )
            .unwrap();
        let response = stream.finish("fallback").unwrap();
        assert_eq!(response.message.text(), "hello");
        assert_eq!(
            response.message.tool_calls[0].arguments,
            json!({ "text": "hi" })
        );
        assert_eq!(response.finish_reason, FinishReason::ToolCalls);
        assert_eq!(response.usage.output_tokens, 4);
        assert_eq!(deltas.lock().unwrap().len(), 3);
    }
}
