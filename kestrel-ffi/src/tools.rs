use std::sync::Arc;

use async_trait::async_trait;
use kestrel::{Error, Result, Tool, ToolContext, ToolOutput, ToolSpec};
use serde_json::Value;

use crate::callbacks::{CallbackTracker, ToolCallback};

pub(crate) struct CallbackTool {
    spec: ToolSpec,
    callback: ToolCallback,
    tracker: Arc<CallbackTracker>,
}

impl CallbackTool {
    pub(crate) fn new(
        spec: ToolSpec,
        callback: ToolCallback,
        tracker: Arc<CallbackTracker>,
    ) -> Self {
        Self {
            spec,
            callback,
            tracker,
        }
    }
}

#[async_trait]
impl Tool for CallbackTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, arguments: Value, context: &ToolContext) -> Result<ToolOutput> {
        let arguments_json = serde_json::to_string(&arguments)?;
        let call_id = context.call_id.clone();
        let callback = self.callback;
        let active = self.tracker.enter();
        let response = tokio::task::spawn_blocking(move || {
            let _active = active;
            callback.invoke(&call_id, &arguments_json)
        })
        .await
        .map_err(|error| {
            Error::tool(
                &self.spec.name,
                format!("tool callback task failed: {error}"),
            )
        })?
        .map_err(|error| Error::tool(&self.spec.name, error))?;

        match serde_json::from_str::<ToolOutput>(&response) {
            Ok(output) => Ok(output),
            Err(_) => Ok(ToolOutput::new(response)),
        }
    }
}
