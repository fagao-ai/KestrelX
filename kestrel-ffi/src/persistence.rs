use std::sync::Arc;

use async_trait::async_trait;
use kestrel::{Error, Memory, MemoryEntry, MemoryQuery, Result, Session, SessionStore};

use crate::callbacks::{
    CallbackTracker, MemoryRecallCallback, MemoryRememberCallback, SessionDeleteCallback,
    SessionLoadCallback, SessionSaveCallback,
};

pub(crate) struct CallbackSessionStore {
    load: SessionLoadCallback,
    save: SessionSaveCallback,
    delete: Option<SessionDeleteCallback>,
    tracker: Arc<CallbackTracker>,
}

impl CallbackSessionStore {
    pub(crate) fn new(
        load: SessionLoadCallback,
        save: SessionSaveCallback,
        delete: Option<SessionDeleteCallback>,
        tracker: Arc<CallbackTracker>,
    ) -> Self {
        Self {
            load,
            save,
            delete,
            tracker,
        }
    }
}

#[async_trait]
impl SessionStore for CallbackSessionStore {
    async fn load(&self, id: &str) -> Result<Option<Session>> {
        let id = id.to_owned();
        let load = self.load;
        let active = self.tracker.enter();
        let result = tokio::task::spawn_blocking(move || {
            let _active = active;
            load.invoke(&id)
        })
        .await
        .map_err(|error| Error::Protocol(format!("session load task failed: {error}")))?
        .map_err(Error::Protocol)?;

        result
            .map(|json| {
                serde_json::from_str(&json).map_err(|error| {
                    Error::Protocol(format!("invalid session JSON from host: {error}"))
                })
            })
            .transpose()
    }

    async fn save(&self, session: &Session) -> Result<()> {
        let session_json = serde_json::to_string(session)?;
        let save = self.save;
        let active = self.tracker.enter();
        tokio::task::spawn_blocking(move || {
            let _active = active;
            save.invoke(&session_json)
        })
        .await
        .map_err(|error| Error::Protocol(format!("session save task failed: {error}")))?
        .map_err(Error::Protocol)
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let Some(delete) = self.delete else {
            return Ok(());
        };
        let id = id.to_owned();
        let active = self.tracker.enter();
        tokio::task::spawn_blocking(move || {
            let _active = active;
            delete.invoke(&id)
        })
        .await
        .map_err(|error| Error::Protocol(format!("session delete task failed: {error}")))?
        .map_err(Error::Protocol)
    }
}

pub(crate) struct CallbackMemory {
    recall: MemoryRecallCallback,
    remember: MemoryRememberCallback,
    tracker: Arc<CallbackTracker>,
}

impl CallbackMemory {
    pub(crate) fn new(
        recall: MemoryRecallCallback,
        remember: MemoryRememberCallback,
        tracker: Arc<CallbackTracker>,
    ) -> Self {
        Self {
            recall,
            remember,
            tracker,
        }
    }
}

#[async_trait]
impl Memory for CallbackMemory {
    async fn remember(&self, entry: MemoryEntry) -> Result<()> {
        let entry_json = serde_json::to_string(&entry)?;
        let remember = self.remember;
        let active = self.tracker.enter();
        tokio::task::spawn_blocking(move || {
            let _active = active;
            remember.invoke(&entry_json)
        })
        .await
        .map_err(|error| Error::Memory(format!("memory remember task failed: {error}")))?
        .map_err(Error::Memory)
    }

    async fn recall(&self, query: &MemoryQuery) -> Result<Vec<MemoryEntry>> {
        let query_json = serde_json::to_string(query)?;
        let recall = self.recall;
        let active = self.tracker.enter();
        let result = tokio::task::spawn_blocking(move || {
            let _active = active;
            recall.invoke(&query_json)
        })
        .await
        .map_err(|error| Error::Memory(format!("memory recall task failed: {error}")))?
        .map_err(Error::Memory)?;

        serde_json::from_str(&result)
            .map_err(|error| Error::Memory(format!("invalid memory JSON from host: {error}")))
    }
}
