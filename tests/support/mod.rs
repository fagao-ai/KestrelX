#![allow(dead_code)]

use std::{
    collections::VecDeque,
    future::pending,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use kestrel::{FinishReason, Message, Model, ModelRequest, ModelResponse, Result, ToolCall, Usage};
use serde_json::{Value, json};
use tokio::sync::Notify;

#[derive(Clone, Default)]
pub struct ScriptedModel {
    responses: Arc<Mutex<VecDeque<ModelResponse>>>,
    requests: Arc<Mutex<Vec<ModelRequest>>>,
    context_window: Option<usize>,
}

impl ScriptedModel {
    pub fn new(responses: impl IntoIterator<Item = ModelResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses.into_iter().collect())),
            requests: Arc::new(Mutex::new(Vec::new())),
            context_window: None,
        }
    }

    pub fn with_context_window(mut self, context_window: usize) -> Self {
        self.context_window = Some(context_window);
        self
    }

    pub fn requests(&self) -> Vec<ModelRequest> {
        self.requests.lock().unwrap().clone()
    }

    pub fn remaining(&self) -> usize {
        self.responses.lock().unwrap().len()
    }
}

#[async_trait]
impl Model for ScriptedModel {
    fn name(&self) -> &str {
        "scripted"
    }

    fn context_window(&self) -> Option<usize> {
        self.context_window
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse> {
        self.requests.lock().unwrap().push(request);
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| kestrel::Error::Model("no scripted response left".to_owned()))
    }
}

pub fn text_response(text: impl Into<String>) -> ModelResponse {
    ModelResponse {
        message: Message::assistant(text),
        finish_reason: FinishReason::Stop,
        usage: Usage {
            input_tokens: 10,
            output_tokens: 5,
        },
        model: Some("scripted".to_owned()),
        raw: None,
    }
}

pub fn tool_response(calls: Vec<ToolCall>) -> ModelResponse {
    ModelResponse {
        message: Message::assistant_with_tool_calls("", calls),
        finish_reason: FinishReason::ToolCalls,
        usage: Usage {
            input_tokens: 12,
            output_tokens: 4,
        },
        model: Some("scripted".to_owned()),
        raw: None,
    }
}

pub fn call(id: &str, name: &str, arguments: Value) -> ToolCall {
    ToolCall {
        id: id.to_owned(),
        name: name.to_owned(),
        arguments,
    }
}

pub fn object_schema() -> Value {
    json!({
        "type": "object",
        "properties": {},
        "additionalProperties": false
    })
}

pub struct BlockingModel {
    started: Arc<Notify>,
}

impl BlockingModel {
    pub fn new() -> Self {
        Self {
            started: Arc::new(Notify::new()),
        }
    }

    pub fn started(&self) -> Arc<Notify> {
        self.started.clone()
    }
}

#[async_trait]
impl Model for BlockingModel {
    fn name(&self) -> &str {
        "blocking"
    }

    async fn complete(&self, _request: ModelRequest) -> Result<ModelResponse> {
        self.started.notify_one();
        pending().await
    }
}

#[derive(Clone)]
pub struct StreamingModel {
    delta_chunks: Vec<String>,
    response: ModelResponse,
}

impl StreamingModel {
    pub fn new(
        delta_chunks: impl IntoIterator<Item = impl Into<String>>,
        response: ModelResponse,
    ) -> Self {
        Self {
            delta_chunks: delta_chunks.into_iter().map(Into::into).collect(),
            response,
        }
    }
}

#[async_trait]
impl Model for StreamingModel {
    fn name(&self) -> &str {
        "streaming"
    }

    async fn complete(&self, _request: ModelRequest) -> Result<ModelResponse> {
        Ok(self.response.clone())
    }

    async fn complete_stream(
        &self,
        _request: ModelRequest,
        on_delta: &(dyn Fn(kestrel::ModelDelta) + Send + Sync),
    ) -> Result<ModelResponse> {
        for text in &self.delta_chunks {
            on_delta(kestrel::ModelDelta::Text { text: text.clone() });
        }
        Ok(self.response.clone())
    }
}
