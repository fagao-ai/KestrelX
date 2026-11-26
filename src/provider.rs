use reqwest::{Response, StatusCode};
use serde_json::Value;

use crate::{Error, Result};

pub(crate) fn provider_http_error(status: StatusCode, body: &[u8]) -> Error {
    Error::Provider {
        status: status.as_u16(),
        message: truncate(&String::from_utf8_lossy(body), 2_000),
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
}
