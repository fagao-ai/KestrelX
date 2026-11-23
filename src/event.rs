use serde::Serialize;
use tokio::sync::mpsc;

use crate::{FinishReason, ModelDelta, ToolCall, ToolOutput, Usage};

/// An observer for agent lifecycle events.
///
/// Implementations must be fast and must not await. Forward events to a
/// channel, logger, or SDK boundary and return immediately.
pub trait EventSink: Send + Sync {
    /// Receives one event.
    fn emit(&self, event: AgentEvent);
}

impl EventSink for mpsc::UnboundedSender<AgentEvent> {
    fn emit(&self, event: AgentEvent) {
        let _ = self.send(event);
    }
}

/// Creates an unbounded event channel.
pub fn event_channel() -> (
    mpsc::UnboundedSender<AgentEvent>,
    mpsc::UnboundedReceiver<AgentEvent>,
) {
    mpsc::unbounded_channel()
}

/// A lifecycle event tagged with run context.
#[derive(Debug, Clone, Serialize)]
pub struct AgentEvent {
    /// Unique run identifier.
    pub run_id: String,
    /// Persistent conversation identifier.
    pub session_id: String,
    /// Monotonic one-based sequence within the run.
    pub sequence: u64,
    /// Event payload.
    #[serde(flatten)]
    pub kind: AgentEventKind,
}

/// Serializable lifecycle event payload.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEventKind {
    /// A run has begun.
    RunStarted,
    /// A model turn has begun.
    TurnStarted {
        /// One-based turn number.
        turn: usize,
    },
    /// The model returned a message.
    ModelCompleted {
        /// One-based turn number.
        turn: usize,
        /// Token usage for this call.
        usage: Usage,
        /// Stop reason.
        finish_reason: FinishReason,
    },
    /// Incremental model output for token-level rendering.
    ModelDelta {
        /// One-based turn number.
        turn: usize,
        /// Text or tool-call fragment.
        delta: ModelDelta,
    },
    /// Context was compacted before a model call.
    ContextCompressed {
        /// Estimated input tokens before compaction.
        before_tokens: usize,
        /// Estimated input tokens after compaction.
        after_tokens: usize,
        /// Number of original messages removed.
        removed_messages: usize,
        /// Whether a model-generated summary was inserted.
        summarized: bool,
    },
    /// Memory entries were injected into the temporary prompt.
    MemoryRecalled {
        /// Number of entries injected.
        count: usize,
    },
    /// A tool call has begun.
    ToolStarted {
        /// One-based turn number.
        turn: usize,
        /// Complete call requested by the model.
        call: ToolCall,
    },
    /// A tool call has completed.
    ToolCompleted {
        /// One-based turn number.
        turn: usize,
        /// Call identifier.
        call_id: String,
        /// Tool name.
        name: String,
        /// Tool output.
        output: ToolOutput,
    },
    /// The run completed successfully.
    RunCompleted {
        /// Final assistant text.
        text: String,
        /// Total turns used.
        turns: usize,
        /// Total usage across all model calls.
        usage: Usage,
    },
    /// The run failed.
    RunFailed {
        /// Human-readable failure.
        error: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_run_context_with_payload() {
        let event = AgentEvent {
            run_id: "run-1".to_owned(),
            session_id: "session-1".to_owned(),
            sequence: 3,
            kind: AgentEventKind::TurnStarted { turn: 2 },
        };
        let value = serde_json::to_value(event).unwrap();
        assert_eq!(value["run_id"], "run-1");
        assert_eq!(value["session_id"], "session-1");
        assert_eq!(value["sequence"], 3);
        assert_eq!(value["type"], "turn_started");
        assert_eq!(value["turn"], 2);
    }
}
