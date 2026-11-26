use std::{collections::BTreeMap, time::Duration};

use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::oneshot,
};

use kestrel::{Message, Model, ModelDelta, ModelRequest, ToolChoice};

struct CapturedRequest {
    method: String,
    path: String,
    headers: BTreeMap<String, String>,
    body: Value,
}

struct MockResponse {
    content_type: &'static str,
    body: String,
}

impl MockResponse {
    fn json(body: Value) -> Self {
        Self {
            content_type: "application/json",
            body: body.to_string(),
        }
    }

    fn sse(events: impl IntoIterator<Item = Value>) -> Self {
        let mut body = String::new();
        for event in events {
            body.push_str("data: ");
            body.push_str(&event.to_string());
            body.push_str("\n\n");
        }
        Self {
            content_type: "text/event-stream",
            body,
        }
    }
}

async fn serve(response: MockResponse) -> (String, oneshot::Receiver<CapturedRequest>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, receiver) = oneshot::channel();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let header_end = loop {
            let mut chunk = [0u8; 4096];
            let read = socket.read(&mut chunk).await.unwrap();
            assert!(read > 0, "client closed before sending headers");
            request.extend_from_slice(&chunk[..read]);
            if let Some(position) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                break position + 4;
            }
        };

        let headers_text = String::from_utf8_lossy(&request[..header_end]);
        let mut lines = headers_text.split("\r\n");
        let request_line = lines.next().unwrap();
        let mut request_parts = request_line.split_whitespace();
        let method = request_parts.next().unwrap().to_owned();
        let path = request_parts.next().unwrap().to_owned();
        let mut headers = BTreeMap::new();
        for line in lines {
            let Some((name, value)) = line.split_once(':') else {
                continue;
            };
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_owned());
        }
        let content_length = headers
            .get("content-length")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or_default();
        while request.len() < header_end + content_length {
            let mut chunk = [0u8; 4096];
            let read = socket.read(&mut chunk).await.unwrap();
            assert!(read > 0, "client closed before sending the body");
            request.extend_from_slice(&chunk[..read]);
        }
        let body = serde_json::from_slice(&request[header_end..header_end + content_length])
            .unwrap_or(Value::Null);
        sender
            .send(CapturedRequest {
                method,
                path,
                headers,
                body,
            })
            .ok();

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response.content_type,
            response.body.len(),
            response.body
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        socket.shutdown().await.unwrap();
    });
    (format!("http://{address}"), receiver)
}

fn request() -> ModelRequest {
    ModelRequest {
        messages: vec![Message::system("Be concise."), Message::user("hello")],
        tools: Vec::new(),
        tool_choice: ToolChoice::Auto,
        temperature: None,
        max_tokens: None,
    }
}

async fn captured(receiver: oneshot::Receiver<CapturedRequest>) -> CapturedRequest {
    tokio::time::timeout(Duration::from_secs(5), receiver)
        .await
        .expect("mock server timed out")
        .expect("mock server did not capture the request")
}

#[cfg(feature = "openai")]
mod openai {
    use kestrel::OpenAiCompatible;

    use super::*;

    #[tokio::test]
    async fn completes_chat_completion_over_http() {
        let (base_url, captured_request) = serve(MockResponse::json(json!({
            "model": "test-model",
            "choices": [{
                "message": {"content": "hello from openai"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}
        })))
        .await;
        let provider = OpenAiCompatible::builder("test-model")
            .api_key("secret")
            .base_url(format!("{base_url}/v1"))
            .build()
            .unwrap();

        let response = provider.complete(request()).await.unwrap();
        assert_eq!(response.message.text(), "hello from openai");
        assert_eq!(response.usage.total(), 5);

        let request = captured(captured_request).await;
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/v1/chat/completions");
        assert_eq!(request.headers["authorization"], "Bearer secret");
        assert_eq!(request.body["model"], "test-model");
        assert_eq!(request.body["stream"], false);
        assert_eq!(request.body["messages"][1]["content"], "hello");
    }

    #[tokio::test]
    async fn streams_chat_completion_over_http() {
        let (base_url, captured_request) = serve(MockResponse::sse([
            json!({
                "model": "test-model",
                "choices": [{"index": 0, "delta": {"content": "hel"}}]
            }),
            json!({
                "choices": [{
                    "index": 0,
                    "delta": {"content": "lo"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2}
            }),
        ]))
        .await;
        let provider = OpenAiCompatible::builder("test-model")
            .base_url(format!("{base_url}/v1"))
            .build()
            .unwrap();
        let text = std::sync::Mutex::new(String::new());

        let response = provider
            .complete_stream(request(), &|delta| {
                if let ModelDelta::Text { text: delta } = delta {
                    text.lock().unwrap().push_str(&delta);
                }
            })
            .await
            .unwrap();
        assert_eq!(response.message.text(), "hello");
        assert_eq!(*text.lock().unwrap(), "hello");

        let request = captured(captured_request).await;
        assert_eq!(request.path, "/v1/chat/completions");
        assert_eq!(request.body["stream"], true);
    }
}

#[cfg(feature = "openai")]
mod openai_responses {
    use kestrel::OpenAiResponses;

    use super::*;

    #[tokio::test]
    async fn completes_responses_api_over_http() {
        let (base_url, captured_request) = serve(MockResponse::json(json!({
            "model": "test-model",
            "status": "completed",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "hello from responses"}]
            }],
            "usage": {"input_tokens": 3, "output_tokens": 2}
        })))
        .await;
        let provider = OpenAiResponses::builder("test-model")
            .api_key("secret")
            .base_url(format!("{base_url}/v1"))
            .build()
            .unwrap();

        let response = provider.complete(request()).await.unwrap();
        assert_eq!(response.message.text(), "hello from responses");
        assert_eq!(response.usage.total(), 5);

        let request = captured(captured_request).await;
        assert_eq!(request.path, "/v1/responses");
        assert_eq!(request.headers["authorization"], "Bearer secret");
        assert_eq!(request.body["instructions"], "Be concise.");
        assert_eq!(request.body["input"][0]["content"], "hello");
    }

    #[tokio::test]
    async fn streams_responses_api_with_empty_terminal_output() {
        let (base_url, captured_request) = serve(MockResponse::sse([
            json!({"type": "response.output_text.delta", "delta": "hel"}),
            json!({"type": "response.output_text.delta", "delta": "lo"}),
            json!({
                "type": "response.completed",
                "response": {
                    "model": "test-model",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 3, "output_tokens": 2}
                }
            }),
        ]))
        .await;
        let provider = OpenAiResponses::builder("test-model")
            .base_url(format!("{base_url}/v1"))
            .build()
            .unwrap();
        let text = std::sync::Mutex::new(String::new());

        let response = provider
            .complete_stream(request(), &|delta| {
                if let ModelDelta::Text { text: delta } = delta {
                    text.lock().unwrap().push_str(&delta);
                }
            })
            .await
            .unwrap();
        assert_eq!(response.message.text(), "hello");
        assert_eq!(*text.lock().unwrap(), "hello");

        let request = captured(captured_request).await;
        assert_eq!(request.path, "/v1/responses");
        assert_eq!(request.body["stream"], true);
    }
}

#[cfg(feature = "anthropic")]
mod anthropic {
    use kestrel::Anthropic;

    use super::*;

    #[tokio::test]
    async fn completes_anthropic_messages_over_http() {
        let (base_url, captured_request) = serve(MockResponse::json(json!({
            "model": "test-model",
            "content": [{"type": "text", "text": "hello from claude"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 2}
        })))
        .await;
        let provider = Anthropic::builder("test-model")
            .api_key("secret")
            .base_url(format!("{base_url}/v1"))
            .build()
            .unwrap();

        let response = provider.complete(request()).await.unwrap();
        assert_eq!(response.message.text(), "hello from claude");
        assert_eq!(response.usage.total(), 5);

        let request = captured(captured_request).await;
        assert_eq!(request.path, "/v1/messages");
        assert_eq!(request.headers["x-api-key"], "secret");
        assert_eq!(request.headers["anthropic-version"], "2023-06-01");
        assert_eq!(request.body["system"], "Be concise.");
        assert_eq!(request.body["messages"][0]["content"], "hello");
    }

    #[tokio::test]
    async fn streams_anthropic_messages_over_http() {
        let (base_url, captured_request) = serve(MockResponse::sse([
            json!({
                "type": "message_start",
                "message": {"model": "test-model", "usage": {"input_tokens": 3}}
            }),
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            }),
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hel"}
            }),
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "lo"}
            }),
            json!({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 2}
            }),
            json!({"type": "message_stop"}),
        ]))
        .await;
        let provider = Anthropic::builder("test-model")
            .base_url(format!("{base_url}/v1"))
            .build()
            .unwrap();
        let text = std::sync::Mutex::new(String::new());

        let response = provider
            .complete_stream(request(), &|delta| {
                if let ModelDelta::Text { text: delta } = delta {
                    text.lock().unwrap().push_str(&delta);
                }
            })
            .await
            .unwrap();
        assert_eq!(response.message.text(), "hello");
        assert_eq!(*text.lock().unwrap(), "hello");

        let request = captured(captured_request).await;
        assert_eq!(request.path, "/v1/messages");
        assert_eq!(request.body["stream"], true);
    }
}

#[cfg(feature = "gemini")]
mod gemini {
    use kestrel::Gemini;

    use super::*;

    #[tokio::test]
    async fn completes_gemini_generate_content_over_http() {
        let (base_url, captured_request) = serve(MockResponse::json(json!({
            "modelVersion": "test-model",
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "hello from gemini"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2}
        })))
        .await;
        let provider = Gemini::builder("test-model")
            .api_key("secret")
            .base_url(format!("{base_url}/v1beta"))
            .build()
            .unwrap();

        let response = provider.complete(request()).await.unwrap();
        assert_eq!(response.message.text(), "hello from gemini");
        assert_eq!(response.usage.total(), 5);

        let request = captured(captured_request).await;
        assert_eq!(request.path, "/v1beta/models/test-model:generateContent");
        assert_eq!(request.headers["x-goog-api-key"], "secret");
        assert_eq!(
            request.body["systemInstruction"]["parts"][0]["text"],
            "Be concise."
        );
        assert_eq!(request.body["contents"][0]["parts"][0]["text"], "hello");
    }

    #[tokio::test]
    async fn streams_gemini_generate_content_over_http() {
        let (base_url, captured_request) = serve(MockResponse::sse([
            json!({
                "modelVersion": "test-model",
                "candidates": [{"content": {"role": "model", "parts": [{"text": "hel"}]}}]
            }),
            json!({
                "candidates": [{
                    "content": {"role": "model", "parts": [{"text": "lo"}]},
                    "finishReason": "STOP"
                }],
                "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2}
            }),
        ]))
        .await;
        let provider = Gemini::builder("test-model")
            .base_url(format!("{base_url}/v1beta"))
            .build()
            .unwrap();
        let text = std::sync::Mutex::new(String::new());

        let response = provider
            .complete_stream(request(), &|delta| {
                if let ModelDelta::Text { text: delta } = delta {
                    text.lock().unwrap().push_str(&delta);
                }
            })
            .await
            .unwrap();
        assert_eq!(response.message.text(), "hello");
        assert_eq!(*text.lock().unwrap(), "hello");

        let request = captured(captured_request).await;
        assert_eq!(
            request.path,
            "/v1beta/models/test-model:streamGenerateContent?alt=sse"
        );
        assert_eq!(request.body["stream"], Value::Null);
    }
}
