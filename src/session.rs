use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;

use crate::{Error, Message, Result, util::now_ms};

/// A persistent conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
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
            id: id.into(),
            messages: Vec::new(),
            metadata: BTreeMap::new(),
            created_at_ms: timestamp,
            updated_at_ms: timestamp,
        }
    }

    /// Appends one message and updates the timestamp.
    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
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

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

/// Persistence for conversations.
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Loads a session if it exists.
    async fn load(&self, id: &str) -> Result<Option<Session>>;

    /// Creates or replaces a session.
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
        self.sessions
            .write()
            .await
            .insert(session.id.clone(), session.clone());
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
