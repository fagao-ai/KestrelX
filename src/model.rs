use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{FinishReason, Message, Result, ToolSpec, Usage};

/// Incremental content emitted while a model response is being generated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelDelta {
    /// A text fragment. Fragments may be characters, tokens, or larger chunks.
    Text {
        /// Incremental text to append.
        text: String,
    },
    /// A partial tool call. Fields arrive incrementally and may be fragmented.
    ToolCall {
        /// Stable tool-call slot used to merge fragments.
        index: usize,
        /// Provider call ID when present in this fragment.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Tool name when present in this fragment.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// JSON argument fragment, if any.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        arguments: Option<String>,
    },
}

/// How a model should choose tools.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// Let the model decide.
    #[default]
    Auto,
    /// Disable tool calls.
    None,
    /// Require at least one tool call.
    Required,
    /// Require one specific tool.
    Named(String),
}

/// A provider-neutral model request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRequest {
    /// Full prompt, including temporary system messages.
    pub messages: Vec<Message>,
    /// Tools available for this turn.
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    /// Tool selection mode.
    #[serde(default)]
    pub tool_choice: ToolChoice,
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Maximum number of output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

/// A provider-neutral model response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    /// Assistant message. Its role must be [`crate::Role::Assistant`].
    pub message: Message,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
    /// Provider-reported usage.
    #[serde(default)]
    pub usage: Usage,
    /// Concrete model name reported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Unmodified provider payload for debugging and provider-specific data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<serde_json::Value>,
}

/// A language model provider.
///
/// Implementations should be cheap to clone behind an `Arc`. Kestrel never
/// assumes a specific vendor, SDK, or HTTP API.
#[async_trait]
pub trait Model: Send + Sync {
    /// Stable human-readable model name used in diagnostics.
    fn name(&self) -> &str;

    /// Optional context window used by automatic compression.
    fn context_window(&self) -> Option<usize> {
        None
    }

    /// Performs exactly one model completion.
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse>;

    /// Performs one completion while forwarding incremental output.
    ///
    /// The returned response must contain the fully assembled assistant
    /// message. The default implementation calls [`Model::complete`] and
    /// emits the final text as one delta, so providers are not required to
    /// support streaming. `on_delta` is synchronous and must not block.
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

#[async_trait]
impl<T> Model for Arc<T>
where
    T: Model + ?Sized,
{
    fn name(&self) -> &str {
        self.as_ref().name()
    }

    fn context_window(&self) -> Option<usize> {
        self.as_ref().context_window()
    }

    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse> {
        self.as_ref().complete(request).await
    }

    async fn complete_stream(
        &self,
        request: ModelRequest,
        on_delta: &(dyn Fn(ModelDelta) + Send + Sync),
    ) -> Result<ModelResponse> {
        self.as_ref().complete_stream(request, on_delta).await
    }
}

impl std::fmt::Debug for dyn Model {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Model")
            .field("name", &self.name())
            .finish_non_exhaustive()
    }
}
