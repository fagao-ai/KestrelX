use std::sync::Arc;

use async_trait::async_trait;
use kestrel::{Error, Model, ModelDelta, ModelRequest, ModelResponse, Result};

use tokio::sync::mpsc;

use crate::callbacks::{CallbackTracker, ModelCallback, StreamingModelCallback};

pub(crate) struct CallbackModel {
    name: String,
    context_window: Option<usize>,
    backend: CallbackBackend,
    tracker: Arc<CallbackTracker>,
}

enum CallbackBackend {
    Complete(ModelCallback),
    Streaming(StreamingModelCallback),
}

impl CallbackModel {
    pub(crate) fn new(
        name: impl Into<String>,
        context_window: Option<usize>,
        callback: ModelCallback,
        tracker: Arc<CallbackTracker>,
    ) -> Self {
        Self {
            name: name.into(),
            context_window,
            backend: CallbackBackend::Complete(callback),
            tracker,
        }
    }

    pub(crate) fn new_streaming(
        name: impl Into<String>,
        context_window: Option<usize>,
        callback: StreamingModelCallback,
        tracker: Arc<CallbackTracker>,
    ) -> Self {
        Self {
            name: name.into(),
            context_window,
            backend: CallbackBackend::Streaming(callback),
            tracker,
        }
    }

    async fn complete_internal(
        &self,
        request: ModelRequest,
        on_delta: Option<&(dyn Fn(ModelDelta) + Send + Sync)>,
    ) -> Result<ModelResponse> {
        let request_json = serde_json::to_string(&request)?;
        let active = self.tracker.enter();

        let response_json = match &self.backend {
            CallbackBackend::Complete(callback) => {
                let callback = *callback;
                let response_json = tokio::task::spawn_blocking(move || {
                    let _active = active;
                    callback.invoke(&request_json)
                })
                .await
                .map_err(|error| Error::Model(format!("model callback task failed: {error}")))?
                .map_err(Error::Model)?;
                let response: ModelResponse =
                    serde_json::from_str(&response_json).map_err(|error| {
                        Error::InvalidModelResponse(format!(
                            "model callback returned invalid JSON: {error}"
                        ))
                    })?;
                if let Some(on_delta) = on_delta {
                    let text = response.message.text();
                    if !text.is_empty() {
                        on_delta(ModelDelta::Text { text });
                    }
                }
                return Ok(response);
            }
            CallbackBackend::Streaming(callback) => {
                let callback = *callback;
                let (sender, mut receiver) = mpsc::unbounded_channel();
                let mut task = tokio::task::spawn_blocking(move || {
                    let _active = active;
                    callback.invoke(&request_json, sender)
                });
                let mut channel_open = true;
                let response_json = loop {
                    tokio::select! {
                        delta = receiver.recv(), if channel_open => {
                            match delta {
                                Some(delta) => {
                                    if let Some(on_delta) = on_delta {
                                        on_delta(delta);
                                    }
                                }
                                None => channel_open = false,
                            }
                        }
                        result = &mut task => {
                            break result.map_err(|error| {
                                Error::Model(format!("streaming model callback task failed: {error}"))
                            })?.map_err(Error::Model)?;
                        }
                    }
                };
                while let Ok(delta) = receiver.try_recv() {
                    if let Some(on_delta) = on_delta {
                        on_delta(delta);
                    }
                }
                response_json
            }
        };

        serde_json::from_str(&response_json).map_err(|error| {
            Error::InvalidModelResponse(format!("model callback returned invalid JSON: {error}"))
        })
    }
}

#[async_trait]
impl Model for CallbackModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn context_window(&self) -> Option<usize> {
        self.context_window
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse> {
        self.complete_internal(request, None).await
    }

    async fn complete_stream(
        &self,
        request: ModelRequest,
        on_delta: &(dyn Fn(ModelDelta) + Send + Sync),
    ) -> Result<ModelResponse> {
        self.complete_internal(request, Some(on_delta)).await
    }
}
