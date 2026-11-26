use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;

use crate::{Error, Message, Result, Usage, util::now_ms};

/// Current serialized session schema.
pub const SESSION_SCHEMA_VERSION: u32 = 1;

fn default_schema_version() -> u32 {
    SESSION_SCHEMA_VERSION
}

/// A persistent conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Version of the serialized session format.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// Stable conversation identifier.
    pub id: String,
    /// Full conversation history.
    #[serde(default)]
    pub messages: Vec<Message>,
    /// Application-owned metadata.
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
    /// Creation time in Unix milliseconds.
    #[serde(default)]
    pub created_at_ms: u64,
    /// Last update time in Unix milliseconds.
    #[serde(default)]
    pub updated_at_ms: u64,
    /// Monotonic write revision used by host stores for concurrency control.
    #[serde(default)]
    pub revision: u64,
    /// Durable state for an interrupted run, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_run: Option<RunCheckpoint>,
}

impl Session {
    /// Creates an empty session with a random ID.
    pub fn new() -> Self {
        Self::with_id(uuid::Uuid::new_v4().to_string())
    }

    /// Creates an empty session with a caller-supplied ID.
    pub fn with_id(id: impl Into<String>) -> Self {
        let timestamp = now_ms();
        Self {
            schema_version: SESSION_SCHEMA_VERSION,
            id: id.into(),
            messages: Vec::new(),
            metadata: BTreeMap::new(),
            created_at_ms: timestamp,
            updated_at_ms: timestamp,
            revision: 0,
            active_run: None,
        }
    }

    /// Appends one message and updates the timestamp.
    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
        self.updated_at_ms = now_ms();
    }

    /// Marks the session as changed and advances its write revision.
    pub fn touch(&mut self) {
        self.revision = self.revision.saturating_add(1);
        self.updated_at_ms = now_ms();
    }

    /// Returns the latest user message text.
    pub fn last_user_text(&self) -> Option<String> {
        self.messages
            .iter()
            .rev()
            .find(|message| message.role == crate::Role::User)
            .map(Message::text)
    }
}

/// Durable checkpoint for an interrupted run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunCheckpoint {
    /// Identifier of the interrupted run.
    pub run_id: String,
    /// One-based turn to execute next.
    pub turn: usize,
    /// Usage accumulated before this checkpoint.
    #[serde(default)]
    pub usage: Usage,
    /// Next safe operation.
    pub phase: RunPhase,
    /// Last checkpoint update time in Unix milliseconds.
    #[serde(default)]
    pub updated_at_ms: u64,
}

impl RunCheckpoint {
    /// Creates a checkpoint that will perform a model call.
    pub fn model_call(run_id: impl Into<String>, turn: usize, usage: Usage) -> Self {
        Self {
            run_id: run_id.into(),
            turn,
            usage,
            phase: RunPhase::ModelCall,
            updated_at_ms: now_ms(),
        }
    }

    /// Creates a checkpoint after an assistant requested tools.
    pub fn tools(run_id: impl Into<String>, turn: usize, usage: Usage, assistant: Message) -> Self {
        Self {
            run_id: run_id.into(),
            turn,
            usage,
            phase: RunPhase::Tools {
                assistant,
                completed_results: Vec::new(),
                started: false,
            },
            updated_at_ms: now_ms(),
        }
    }

    /// Returns the tool call that should execute next.
    pub fn pending_tool_call(&self) -> Option<&crate::ToolCall> {
        let RunPhase::Tools {
            assistant,
            completed_results,
            ..
        } = &self.phase
        else {
            return None;
        };
        assistant.tool_calls.get(completed_results.len())
    }

    /// Returns whether a tool call was started but its result was not checkpointed.
    pub fn tool_started(&self) -> bool {
        matches!(&self.phase, RunPhase::Tools { started: true, .. })
    }

    /// Marks the pending tool call as started.
    pub fn mark_tool_started(&mut self) {
        if let RunPhase::Tools { started, .. } = &mut self.phase {
            *started = true;
        }
        self.updated_at_ms = now_ms();
    }

    /// Appends one completed tool result.
    pub fn push_tool_result(&mut self, result: Message) {
        if let RunPhase::Tools {
            completed_results,
            started,
            ..
        } = &mut self.phase
        {
            completed_results.push(result);
            *started = false;
        }
        self.updated_at_ms = now_ms();
    }

    /// Returns the assistant message and completed tool results once all tools finish.
    pub fn take_completed_tools(self) -> Option<(Message, Vec<Message>)> {
        let RunPhase::Tools {
            assistant,
            completed_results,
            started,
        } = self.phase
        else {
            return None;
        };
        if started || completed_results.len() != assistant.tool_calls.len() {
            return None;
        }
        Some((assistant, completed_results))
    }

    /// Validates checkpoint invariants before resuming.
    pub fn validate(&self) -> Result<()> {
        if self.run_id.trim().is_empty() {
            return Err(Error::Protocol(
                "run checkpoint has an empty run_id".to_owned(),
            ));
        }
        if self.turn == 0 {
            return Err(Error::Protocol(
                "run checkpoint turn must be at least 1".to_owned(),
            ));
        }
        let RunPhase::Tools {
            assistant,
            completed_results,
            ..
        } = &self.phase
        else {
            return Ok(());
        };
        if assistant.role != crate::Role::Assistant || assistant.tool_calls.is_empty() {
            return Err(Error::Protocol(
                "tool checkpoint must contain an assistant message with tool calls".to_owned(),
            ));
        }
        if completed_results.len() > assistant.tool_calls.len() {
            return Err(Error::Protocol(
                "tool checkpoint contains more results than tool calls".to_owned(),
            ));
        }
        for (call, result) in assistant.tool_calls.iter().zip(completed_results) {
            if result.role != crate::Role::Tool
                || result.tool_call_id.as_deref() != Some(call.id.as_str())
            {
                return Err(Error::Protocol(format!(
                    "tool checkpoint result does not match call `{}`",
                    call.id
                )));
            }
        }
        Ok(())
    }
}

/// The next operation in an interrupted run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RunPhase {
    /// Call the model for the current turn.
    ModelCall,
    /// Execute or finish the assistant's tool calls.
    Tools {
        /// Assistant message containing the complete tool call set.
        assistant: Message,
        /// Results already committed in call order.
        #[serde(default)]
        completed_results: Vec<Message>,
        /// Whether the next tool call was started but not completed.
        #[serde(default)]
        started: bool,
    },
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_legacy_sessions_with_defaults() {
        let session: Session = serde_json::from_value(serde_json::json!({
            "id": "legacy",
            "messages": []
        }))
        .unwrap();
        assert_eq!(session.schema_version, SESSION_SCHEMA_VERSION);
        assert_eq!(session.revision, 0);
        assert!(session.active_run.is_none());
    }

    #[test]
    fn round_trips_tool_checkpoint() {
        let assistant = Message::assistant_with_tool_calls(
            "",
            vec![crate::ToolCall {
                id: "call-1".to_owned(),
                name: "echo".to_owned(),
                arguments: serde_json::json!({ "text": "hello" }),
            }],
        );
        let mut checkpoint = RunCheckpoint::tools("run-1", 2, Usage::default(), assistant);
        checkpoint.mark_tool_started();
        let encoded = serde_json::to_value(&checkpoint).unwrap();
        let decoded: RunCheckpoint = serde_json::from_value(encoded).unwrap();
        assert!(decoded.tool_started());
        assert_eq!(decoded.pending_tool_call().unwrap().id, "call-1");
    }

    #[tokio::test]
    async fn in_memory_store_rejects_stale_revisions() {
        let store = InMemorySessionStore::new();
        let mut session = Session::with_id("conflict");
        session.revision = 2;
        store.save(&session).await.unwrap();

        session.revision = 1;
        assert!(matches!(
            store.save(&session).await,
            Err(Error::SessionConflict {
                stored_revision: 2,
                provided_revision: 1,
                ..
            })
        ));
    }
}

/// Persistence for conversations.
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Loads a session if it exists.
    async fn load(&self, id: &str) -> Result<Option<Session>>;

    /// Creates or replaces a session.
    ///
    /// Implementations should persist the complete serializable session and
    /// may use [`Session::revision`] for optimistic concurrency control.
    async fn save(&self, session: &Session) -> Result<()>;

    /// Deletes a session. The default implementation does nothing.
    async fn delete(&self, _id: &str) -> Result<()> {
        Ok(())
    }
}

/// Process-local session storage.
#[derive(Debug, Clone, Default)]
pub struct InMemorySessionStore {
    sessions: Arc<RwLock<HashMap<String, Session>>>,
}

impl InMemorySessionStore {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of stored sessions.
    pub async fn len(&self) -> usize {
        self.sessions.read().await.len()
    }

    /// Returns whether the store is empty.
    pub async fn is_empty(&self) -> bool {
        self.sessions.read().await.is_empty()
    }
}

#[async_trait]
impl SessionStore for InMemorySessionStore {
    async fn load(&self, id: &str) -> Result<Option<Session>> {
        Ok(self.sessions.read().await.get(id).cloned())
    }

    async fn save(&self, session: &Session) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        if let Some(stored) = sessions.get(&session.id)
            && session.revision < stored.revision
        {
            return Err(Error::SessionConflict {
                id: session.id.clone(),
                stored_revision: stored.revision,
                provided_revision: session.revision,
            });
        }
        sessions.insert(session.id.clone(), session.clone());
        Ok(())
    }

    async fn delete(&self, id: &str) -> Result<()> {
        self.sessions.write().await.remove(id);
        Ok(())
    }
}

impl InMemorySessionStore {
    /// Loads a session or returns [`Error::SessionNotFound`].
    pub async fn require(&self, id: &str) -> Result<Session> {
        self.load(id)
            .await?
            .ok_or_else(|| Error::SessionNotFound(id.to_owned()))
    }
}
