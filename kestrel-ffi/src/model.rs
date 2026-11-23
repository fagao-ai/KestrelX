use std::sync::Arc;

use async_trait::async_trait;
use kestrel::{Error, Model, ModelDelta, ModelRequest, ModelResponse, Result};

use crate::callbacks::{CallbackTracker, ModelCallback};

pub(crate) struct CallbackModel {
    name: String,
    context_window: Option<usize>,
    callback: ModelCallback,
    tracker: Arc<CallbackTracker>,
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
            callback,
            tracker,
        }
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
        let request_json = serde_json::to_string(&request)?;
        let callback = self.callback;
        let active = self.tracker.enter();
        let response_json = tokio::task::spawn_blocking(move || {
            let _active = active;
            callback.invoke(&request_json)
        })
        .await
        .map_err(|error| Error::Model(format!("model callback task failed: {error}")))?
        .map_err(Error::Model)?;
        serde_json::from_str(&response_json).map_err(|error| {
            Error::InvalidModelResponse(format!("model callback returned invalid JSON: {error}"))
        })
    }

    async fn complete_stream(
        &self,
        request: ModelRequest,
        on_delta: &(dyn Fn(ModelDelta) + Send + Sync),
    ) -> Result<ModelResponse> {
        let response = self.complete(request).await?;
        let text = response.message.text();
        if !text.is_empty() {
            on_delta(ModelDelta::Text { text });
        }
        Ok(response)
    }
}
