//! SQLite adapter for Kestrel's [`SessionStore`].
//!
//! The adapter stores each [`Session`] as versioned JSON and keeps its own
//! database schema version in `PRAGMA user_version`. Callers never need to run
//! migrations manually.

use std::{
    fmt::Display,
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};

use async_trait::async_trait;
use kestrel::{Error, Result, Session, SessionStore};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};

/// Current SQLite store schema version.
pub const STORE_SCHEMA_VERSION: i64 = 1;

const CREATE_SCHEMA: &str = r#"
CREATE TABLE kestrel_sessions (
    id TEXT PRIMARY KEY NOT NULL,
    revision INTEGER NOT NULL CHECK (revision >= 0),
    updated_at_ms INTEGER NOT NULL,
    schema_version INTEGER NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX kestrel_sessions_updated_at
    ON kestrel_sessions(updated_at_ms DESC);
"#;

/// A durable [`SessionStore`] backed by SQLite.
#[derive(Clone)]
pub struct SqliteSessionStore {
    connection: Arc<Mutex<Connection>>,
}

impl std::fmt::Debug for SqliteSessionStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SqliteSessionStore")
            .finish_non_exhaustive()
    }
}

impl SqliteSessionStore {
    /// Opens or creates a SQLite database and migrates its schema.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_connection(Connection::open(path).map_err(sqlite_error)?)
    }

    /// Creates an in-memory store. Each clone shares the same database.
    pub fn open_in_memory() -> Result<Self> {
        Self::from_connection(Connection::open_in_memory().map_err(sqlite_error)?)
    }

    fn from_connection(mut connection: Connection) -> Result<Self> {
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(sqlite_error)?;
        connection
            .execute_batch(
                "PRAGMA foreign_keys = ON;
                 PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = NORMAL;",
            )
            .map_err(sqlite_error)?;
        migrate(&mut connection)?;
        Ok(Self {
            connection: Arc::new(Mutex::new(connection)),
        })
    }

    /// Returns the adapter's database schema version.
    pub async fn schema_version(&self) -> Result<i64> {
        self.run(|connection| {
            connection
                .query_row("PRAGMA user_version", [], |row| row.get(0))
                .map_err(sqlite_error)
        })
        .await
    }

    async fn run<T, F>(&self, operation: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut Connection) -> Result<T> + Send + 'static,
    {
        let connection = self.connection.clone();
        tokio::task::spawn_blocking(move || {
            let mut connection = connection.lock().unwrap_or_else(|error| error.into_inner());
            operation(&mut connection)
        })
        .await
        .map_err(|error| Error::Storage(format!("SQLite task failed: {error}")))?
    }
}

#[async_trait]
impl SessionStore for SqliteSessionStore {
    async fn load(&self, id: &str) -> Result<Option<Session>> {
        let id = id.to_owned();
        self.run(move |connection| {
            let payload: Option<String> = connection
                .query_row(
                    "SELECT payload FROM kestrel_sessions WHERE id = ?1",
                    params![id],
                    |row| row.get(0),
                )
                .optional()
                .map_err(sqlite_error)?;
            let Some(payload) = payload else {
                return Ok(None);
            };
            let session: Session = serde_json::from_str(&payload)?;
            if session.id != id {
                return Err(Error::Storage(format!(
                    "SQLite session payload id `{}` does not match row id `{id}`",
                    session.id
                )));
            }
            Ok(Some(session))
        })
        .await
    }

    async fn save(&self, session: &Session) -> Result<()> {
        let id = session.id.clone();
        let revision = i64::try_from(session.revision).map_err(|_| {
            Error::Storage(format!(
                "session revision {} exceeds SQLite INTEGER range",
                session.revision
            ))
        })?;
        let updated_at_ms = i64::try_from(session.updated_at_ms).unwrap_or(i64::MAX);
        let schema_version = i64::from(session.schema_version);
        let payload = serde_json::to_string(session)?;

        self.run(move |connection| {
            let transaction = connection
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(sqlite_error)?;
            let stored_revision: Option<i64> = transaction
                .query_row(
                    "SELECT revision FROM kestrel_sessions WHERE id = ?1",
                    params![id],
                    |row| row.get(0),
                )
                .optional()
                .map_err(sqlite_error)?;
            if let Some(stored_revision) = stored_revision
                && revision < stored_revision
            {
                return Err(Error::SessionConflict {
                    id,
                    stored_revision: u64::try_from(stored_revision).unwrap_or(u64::MAX),
                    provided_revision: u64::try_from(revision).unwrap_or(u64::MAX),
                });
            }
            transaction
                .execute(
                    "INSERT INTO kestrel_sessions
                        (id, revision, updated_at_ms, schema_version, payload)
                     VALUES (?1, ?2, ?3, ?4, ?5)
                     ON CONFLICT(id) DO UPDATE SET
                        revision = excluded.revision,
                        updated_at_ms = excluded.updated_at_ms,
                        schema_version = excluded.schema_version,
                        payload = excluded.payload",
                    params![id, revision, updated_at_ms, schema_version, payload],
                )
                .map_err(sqlite_error)?;
            transaction.commit().map_err(sqlite_error)
        })
        .await
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let id = id.to_owned();
        self.run(move |connection| {
            connection
                .execute("DELETE FROM kestrel_sessions WHERE id = ?1", params![id])
                .map_err(sqlite_error)?;
            Ok(())
        })
        .await
    }
}

fn migrate(connection: &mut Connection) -> Result<()> {
    let version: i64 = connection
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(sqlite_error)?;
    if version > STORE_SCHEMA_VERSION {
        return Err(Error::Storage(format!(
            "database schema version {version} is newer than supported version {STORE_SCHEMA_VERSION}"
        )));
    }
    if version == 0 {
        let transaction = connection.transaction().map_err(sqlite_error)?;
        transaction
            .execute_batch(CREATE_SCHEMA)
            .map_err(sqlite_error)?;
        transaction
            .pragma_update(None, "user_version", STORE_SCHEMA_VERSION)
            .map_err(sqlite_error)?;
        transaction.commit().map_err(sqlite_error)?;
    }
    Ok(())
}

fn sqlite_error(error: impl Display) -> Error {
    Error::Storage(format!("SQLite: {error}"))
}

#[cfg(test)]
mod tests {
    use kestrel::Message;
    use tempfile::tempdir;

    use super::*;

    fn session(id: &str, text: &str, revision: u64) -> Session {
        let mut session = Session::with_id(id);
        session.push(Message::user(text));
        session.revision = revision;
        session
    }

    #[tokio::test]
    async fn persists_sessions_and_rejects_stale_revisions() {
        let store = SqliteSessionStore::open_in_memory().unwrap();
        let mut value = session("session-1", "hello", 1);
        store.save(&value).await.unwrap();

        let loaded = store.load("session-1").await.unwrap().unwrap();
        assert_eq!(loaded.messages[0].text(), "hello");
        assert_eq!(loaded.revision, 1);

        value.messages[0] = Message::user("updated");
        value.revision = 2;
        store.save(&value).await.unwrap();
        assert_eq!(
            store.load("session-1").await.unwrap().unwrap().messages[0].text(),
            "updated"
        );

        let stale = session("session-1", "stale", 1);
        assert!(matches!(
            store.save(&stale).await,
            Err(Error::SessionConflict {
                stored_revision: 2,
                provided_revision: 1,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn reopens_a_file_store_and_deletes_sessions() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("sessions.sqlite");
        {
            let store = SqliteSessionStore::open(&path).unwrap();
            store.save(&session("persisted", "hello", 1)).await.unwrap();
            assert_eq!(store.schema_version().await.unwrap(), STORE_SCHEMA_VERSION);
        }

        let store = SqliteSessionStore::open(&path).unwrap();
        assert!(store.load("persisted").await.unwrap().is_some());
        store.delete("persisted").await.unwrap();
        assert!(store.load("persisted").await.unwrap().is_none());
    }

    #[test]
    fn rejects_newer_database_schema() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("future.sqlite");
        {
            let connection = Connection::open(&path).unwrap();
            connection
                .pragma_update(None, "user_version", STORE_SCHEMA_VERSION + 1)
                .unwrap();
        }
        assert!(SqliteSessionStore::open(path).is_err());
    }
}
