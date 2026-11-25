use std::result;

/// Errors returned by Kestrel.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The agent or one of its dependencies was configured incorrectly.
    #[error("configuration error: {0}")]
    Configuration(String),

    /// A model invocation failed.
    #[error("model error: {0}")]
    Model(String),

    /// A remote model provider returned a non-successful HTTP response.
    #[error("provider returned HTTP {status}: {message}")]
    Provider {
        /// HTTP status code.
        status: u16,
        /// Response summary.
        message: String,
    },

    /// A named tool failed.
    #[error("tool `{tool}` failed: {message}")]
    Tool {
        /// Tool name.
        tool: String,
        /// Error description.
        message: String,
    },

    /// The model requested a tool that is not registered.
    #[error("unknown tool `{0}`")]
    UnknownTool(String),

    /// The model produced a malformed tool call or response.
    #[error("invalid model response: {0}")]
    InvalidModelResponse(String),

    /// Context could not fit inside the configured budget.
    #[error("context error: {0}")]
    Context(String),

    /// A session could not be found.
    #[error("session `{0}` not found")]
    SessionNotFound(String),

    /// The session contains an interrupted run that must be resumed or reset.
    #[error("session has active run `{0}`; call resume() or reset() first")]
    ActiveRun(String),

    /// There is no interrupted run to resume.
    #[error("session has no active run to resume")]
    NoActiveRun,

    /// A tool call was started but its result was not checkpointed.
    #[error(
        "cannot safely resume tool `{tool}` (call `{call_id}`) because it may have already run"
    )]
    UnsafeResume {
        /// Tool name.
        tool: String,
        /// Tool call identifier.
        call_id: String,
    },

    /// A persisted session uses an unsupported schema version.
    #[error("unsupported session schema version {found}; supported version is {supported}")]
    UnsupportedSessionVersion {
        /// Version found in storage.
        found: u32,
        /// Version supported by this binary.
        supported: u32,
    },

    /// Long-term memory failed.
    #[error("memory error: {0}")]
    Memory(String),

    /// An extension protocol returned an invalid response.
    #[error("protocol error: {0}")]
    Protocol(String),

    /// The run was cancelled by its caller.
    #[error("agent run cancelled")]
    Cancelled,

    /// The loop reached its turn limit with pending tool calls.
    #[error("turn limit of {0} reached before a final response")]
    TurnLimit(usize),

    /// JSON serialization or deserialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),

    /// Local I/O failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// A request to a model provider failed.
    #[cfg(any(feature = "mcp", feature = "openai"))]
    #[error(transparent)]
    Http(#[from] reqwest::Error),
}

impl Error {
    /// Returns whether this error represents cooperative cancellation.
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }

    /// Creates a named tool error for use by custom tool implementations.
    pub fn tool(tool: impl Into<String>, error: impl std::fmt::Display) -> Self {
        Self::Tool {
            tool: tool.into(),
            message: error.to_string(),
        }
    }
}

/// Result alias used throughout Kestrel.
pub type Result<T> = result::Result<T, Error>;
