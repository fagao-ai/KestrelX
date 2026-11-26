use std::{collections::BTreeMap, time::Duration};

use reqwest::{
    Client, Response, StatusCode,
    header::{HeaderMap, HeaderName, HeaderValue},
};
use serde_json::Value;

use crate::{Error, Result};

pub(crate) fn provider_http_error(status: StatusCode, body: &[u8]) -> Error {
    let body = String::from_utf8_lossy(body);
    Error::Provider {
        status: status.as_u16(),
        message: provider_error_message(&body).unwrap_or_else(|| truncate(&body, 2_000)),
    }
}

fn provider_error_message(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    let error = value.get("error").unwrap_or(&value);
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| error.as_str())?;
    let code = error
        .get("code")
        .or_else(|| error.get("type"))
        .and_then(Value::as_str)
        .filter(|code| !code.is_empty());
    Some(match code {
        Some(code) => truncate(&format!("{code}: {message}"), 2_000),
        None => truncate(message, 2_000),
    })
}

pub(crate) fn validate_provider_config(model: &str, base_url: &str) -> Result<()> {
    if model.trim().is_empty() {
        return Err(Error::Configuration(
            "model name cannot be empty".to_owned(),
        ));
    }
    if base_url.trim().is_empty() {
        return Err(Error::Configuration("base URL cannot be empty".to_owned()));
    }
    Ok(())
}

pub(crate) fn insert_extra_headers(
    headers: &mut HeaderMap,
    values: BTreeMap<String, String>,
) -> Result<()> {
    for (name, value) in values {
        let name = HeaderName::from_bytes(name.as_bytes()).map_err(|error| {
            Error::Configuration(format!("invalid HTTP header name `{name}`: {error}"))
        })?;
        let value = HeaderValue::from_str(&value).map_err(|error| {
            Error::Configuration(format!("invalid HTTP header value for `{name}`: {error}"))
        })?;
        headers.insert(name, value);
    }
    Ok(())
}

pub(crate) fn build_http_client(headers: HeaderMap, timeout: Duration) -> Result<Client> {
    Ok(Client::builder()
        .default_headers(headers)
        .timeout(timeout)
        .build()?)
}

#[cfg(any(feature = "openai", test))]
pub(crate) fn resolve_endpoint(
    base_url: &str,
    suffix: &str,
    accepted_full_endpoints: &[&str],
) -> String {
    let base = base_url.trim_end_matches('/');
    if accepted_full_endpoints
        .iter()
        .any(|endpoint| base.ends_with(endpoint))
    {
        base.to_owned()
    } else {
        format!("{base}{suffix}")
    }
}

#[cfg(any(feature = "anthropic", test))]
pub(crate) fn resolve_versioned_endpoint(
    base_url: &str,
    version: &str,
    suffix: &str,
    accepted_full_endpoints: &[&str],
) -> String {
    let base = base_url.trim_end_matches('/');
    if accepted_full_endpoints
        .iter()
        .any(|endpoint| base.ends_with(endpoint))
    {
        base.to_owned()
    } else if base.ends_with(version) {
        format!("{base}{suffix}")
    } else {
        format!("{base}{version}{suffix}")
    }
}

#[cfg(any(feature = "anthropic", feature = "gemini", test))]
pub(crate) fn parse_data_url(url: &str) -> Option<(String, String)> {
    let (metadata, data) = url.strip_prefix("data:")?.split_once(',')?;
    let mime_type = metadata.strip_suffix(";base64")?;
    if mime_type.is_empty() || data.is_empty() {
        return None;
    }
    Some((mime_type.to_owned(), data.to_owned()))
}

#[cfg(any(feature = "openai", test))]
pub(crate) fn decode_tool_arguments(raw: &str) -> Value {
    if raw.trim().is_empty() {
        return Value::Object(Default::default());
    }
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned()))
}

#[cfg(any(feature = "anthropic", test))]
pub(crate) fn object_tool_arguments(arguments: Value) -> Value {
    if arguments.is_null() {
        Value::Object(Default::default())
    } else {
        arguments
    }
}

pub(crate) async fn ensure_success(response: Response) -> Result<Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let body = response.bytes().await?;
    Err(provider_http_error(status, &body))
}

pub(crate) async fn json_response(response: Response) -> Result<Value> {
    let status = response.status();
    let bytes = response.bytes().await?;
    if !status.is_success() {
        return Err(provider_http_error(status, &bytes));
    }
    Ok(serde_json::from_slice(&bytes)?)
}

pub(crate) async fn read_sse(
    mut response: Response,
    mut on_data: impl FnMut(&str) -> Result<bool>,
) -> Result<()> {
    let mut buffer = Vec::new();
    loop {
        let Some(chunk) = response.chunk().await? else {
            break;
        };
        buffer.extend_from_slice(&chunk);
        for data in drain_sse_data(&mut buffer)? {
            if on_data(&data)? {
                return Ok(());
            }
        }
    }
    if !buffer.is_empty() {
        buffer.push(b'\n');
        for data in drain_sse_data(&mut buffer)? {
            if on_data(&data)? {
                return Ok(());
            }
        }
    }
    Ok(())
}

pub(crate) fn drain_sse_data(buffer: &mut Vec<u8>) -> Result<Vec<String>> {
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
                    Error::Protocol(format!("invalid UTF-8 in SSE stream: {error}"))
                })?;
                payloads.push(payload.to_owned());
            }
        }
        consumed = end + 1;
    }

    buffer.drain(..consumed);
    Ok(payloads)
}

pub(crate) fn truncate(value: &str, max_chars: usize) -> String {
    let mut truncated = value.chars().take(max_chars).collect::<String>();
    if value.chars().count() > max_chars {
        truncated.push_str("...");
    }
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_sse_data_across_chunks() {
        let mut buffer = b"event: message\ndata: {\"choices\":[]}\n\ndata: [DONE]\r\n".to_vec();
        let payloads = drain_sse_data(&mut buffer).unwrap();
        assert_eq!(payloads, vec!["{\"choices\":[]}", "[DONE]"]);
        assert!(buffer.is_empty());
    }

    #[test]
    fn extracts_provider_error_message() {
        let body = br#"{"error":{"type":"invalid_request_error","message":"bad model"}}"#;
        let error = provider_http_error(StatusCode::BAD_REQUEST, body);
        assert!(
            error
                .to_string()
                .contains("invalid_request_error: bad model")
        );
    }

    #[test]
    fn resolves_versioned_endpoints() {
        assert_eq!(
            resolve_versioned_endpoint(
                "https://api.example.com",
                "/v1",
                "/messages",
                &["/messages"]
            ),
            "https://api.example.com/v1/messages"
        );
        assert_eq!(
            resolve_versioned_endpoint(
                "https://api.example.com/v1",
                "/v1",
                "/messages",
                &["/messages"]
            ),
            "https://api.example.com/v1/messages"
        );
        assert_eq!(
            resolve_versioned_endpoint(
                "https://api.example.com/v1/messages",
                "/v1",
                "/messages",
                &["/messages"]
            ),
            "https://api.example.com/v1/messages"
        );
    }

    #[test]
    fn parses_media_and_tool_arguments() {
        assert_eq!(
            parse_data_url("data:image/png;base64,aGVsbG8="),
            Some(("image/png".to_owned(), "aGVsbG8=".to_owned()))
        );
        assert!(parse_data_url("https://example.com/image.png").is_none());
        assert_eq!(decode_tool_arguments(""), Value::Object(Default::default()));
        assert_eq!(
            decode_tool_arguments(r#"{"text":"hi"}"#),
            serde_json::json!({"text": "hi"})
        );
        assert_eq!(
            object_tool_arguments(Value::Null),
            Value::Object(Default::default())
        );
    }
}
