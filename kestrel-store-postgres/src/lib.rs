//! PostgreSQL adapter for Kestrel's [`SessionStore`].
//!
//! Sessions are stored as versioned JSON. The adapter owns its table schema and
//! schema version, and uses PostgreSQL advisory locks to make revision checks
//! and writes atomic per session.

use std::time::Duration;

use async_trait::async_trait;
use kestrel::{Error, Result, Session, SessionStore};
use sqlx::{PgPool, postgres::PgPoolOptions};

/// Current PostgreSQL store schema version.
pub const STORE_SCHEMA_VERSION: i64 = 1;

/// A durable [`SessionStore`] backed by PostgreSQL.
#[derive(Clone)]
pub struct PostgresSessionStore {
    pool: PgPool,
}

impl std::fmt::Debug for PostgresSessionStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PostgresSessionStore")
            .finish_non_exhaustive()
    }
}

impl PostgresSessionStore {
    /// Connects to PostgreSQL, creates a small pool, and migrates the schema.
    pub async fn connect(database_url: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .acquire_timeout(Duration::from_secs(5))
            .connect(database_url)
            .await
            .map_err(postgres_error)?;
        Self::connect_with(pool).await
    }

    /// Uses an existing pool and migrates the schema.
    pub async fn connect_with(pool: PgPool) -> Result<Self> {
        migrate(&pool).await?;
        Ok(Self { pool })
    }

    /// Returns the underlying connection pool.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Returns the adapter's database schema version.
    pub async fn schema_version(&self) -> Result<i64> {
        let version: Option<i64> =
            sqlx::query_scalar("SELECT value FROM kestrel_store_meta WHERE key = 'schema_version'")
                .fetch_optional(&self.pool)
                .await
                .map_err(postgres_error)?;
        version
            .ok_or_else(|| Error::Storage("PostgreSQL store schema version is missing".to_owned()))
    }
}

#[async_trait]
impl SessionStore for PostgresSessionStore {
    async fn load(&self, id: &str) -> Result<Option<Session>> {
        let payload: Option<String> =
            sqlx::query_scalar("SELECT payload FROM kestrel_sessions WHERE id = $1")
                .bind(id)
                .fetch_optional(&self.pool)
                .await
                .map_err(postgres_error)?;
        let Some(payload) = payload else {
            return Ok(None);
        };
        let session: Session = serde_json::from_str(&payload)?;
        if session.id != id {
            return Err(Error::Storage(format!(
                "PostgreSQL session payload id `{}` does not match row id `{id}`",
                session.id
            )));
        }
        Ok(Some(session))
    }

    async fn save(&self, session: &Session) -> Result<()> {
        let revision = i64::try_from(session.revision).map_err(|_| {
            Error::Storage(format!(
                "session revision {} exceeds PostgreSQL BIGINT range",
                session.revision
            ))
        })?;
        let updated_at_ms = i64::try_from(session.updated_at_ms).unwrap_or(i64::MAX);
        let schema_version = i64::from(session.schema_version);
        let payload = serde_json::to_string(session)?;

        let mut transaction = self.pool.begin().await.map_err(postgres_error)?;
        sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))")
            .bind(&session.id)
            .execute(&mut *transaction)
            .await
            .map_err(postgres_error)?;
        let stored_revision: Option<i64> =
            sqlx::query_scalar("SELECT revision FROM kestrel_sessions WHERE id = $1")
                .bind(&session.id)
                .fetch_optional(&mut *transaction)
                .await
                .map_err(postgres_error)?;
        if let Some(stored_revision) = stored_revision
            && revision < stored_revision
        {
            return Err(Error::SessionConflict {
                id: session.id.clone(),
                stored_revision: u64::try_from(stored_revision).unwrap_or(u64::MAX),
                provided_revision: u64::try_from(revision).unwrap_or(u64::MAX),
            });
        }
        sqlx::query(
            "INSERT INTO kestrel_sessions
                (id, revision, updated_at_ms, schema_version, payload)
             VALUES ($1, $2, $3, $4, $5)
             ON CONFLICT(id) DO UPDATE SET
                revision = EXCLUDED.revision,
                updated_at_ms = EXCLUDED.updated_at_ms,
                schema_version = EXCLUDED.schema_version,
                payload = EXCLUDED.payload",
        )
        .bind(&session.id)
        .bind(revision)
        .bind(updated_at_ms)
        .bind(schema_version)
        .bind(payload)
        .execute(&mut *transaction)
        .await
        .map_err(postgres_error)?;
        transaction.commit().await.map_err(postgres_error)
    }

    async fn delete(&self, id: &str) -> Result<()> {
        sqlx::query("DELETE FROM kestrel_sessions WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(postgres_error)?;
        Ok(())
    }
}

async fn migrate(pool: &PgPool) -> Result<()> {
    let mut transaction = pool.begin().await.map_err(postgres_error)?;
    sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended('kestrel-store-migrations', 0))")
        .execute(&mut *transaction)
        .await
        .map_err(postgres_error)?;
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS kestrel_store_meta (
            key TEXT PRIMARY KEY,
            value BIGINT NOT NULL
        )",
    )
    .execute(&mut *transaction)
    .await
    .map_err(postgres_error)?;
    let version: Option<i64> =
        sqlx::query_scalar("SELECT value FROM kestrel_store_meta WHERE key = 'schema_version'")
            .fetch_optional(&mut *transaction)
            .await
            .map_err(postgres_error)?;
    let version = version.unwrap_or(0);
    if version > STORE_SCHEMA_VERSION {
        return Err(Error::Storage(format!(
            "database schema version {version} is newer than supported version {STORE_SCHEMA_VERSION}"
        )));
    }
    if version == 0 {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS kestrel_sessions (
                id TEXT PRIMARY KEY,
                revision BIGINT NOT NULL CHECK (revision >= 0),
                updated_at_ms BIGINT NOT NULL,
                schema_version BIGINT NOT NULL,
                payload TEXT NOT NULL
            )",
        )
        .execute(&mut *transaction)
        .await
        .map_err(postgres_error)?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS kestrel_sessions_updated_at
             ON kestrel_sessions(updated_at_ms DESC)",
        )
        .execute(&mut *transaction)
        .await
        .map_err(postgres_error)?;
        sqlx::query(
            "INSERT INTO kestrel_store_meta (key, value)
             VALUES ('schema_version', $1)
             ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value",
        )
        .bind(STORE_SCHEMA_VERSION)
        .execute(&mut *transaction)
        .await
        .map_err(postgres_error)?;
    }
    transaction.commit().await.map_err(postgres_error)
}

fn postgres_error(error: sqlx::Error) -> Error {
    Error::Storage(format!("PostgreSQL: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kestrel::Message;
    use uuid::Uuid;

    fn test_url() -> Option<String> {
        std::env::var("KESTREL_POSTGRES_TEST_URL").ok()
    }

    fn session(id: &str, text: &str, revision: u64) -> Session {
        let mut session = Session::with_id(id);
        session.push(Message::user(text));
        session.revision = revision;
        session
    }

    #[tokio::test]
    async fn persists_sessions_and_rejects_stale_revisions() {
        let Some(url) = test_url() else {
            return;
        };
        let store = PostgresSessionStore::connect(&url).await.unwrap();
        let id = format!("kestrel-test-{}", Uuid::new_v4());
        let mut value = session(&id, "hello", 1);
        store.save(&value).await.unwrap();
        assert_eq!(
            store.load(&id).await.unwrap().unwrap().messages[0].text(),
            "hello"
        );

        value.messages[0] = Message::user("updated");
        value.revision = 2;
        store.save(&value).await.unwrap();
        let stale = session(&id, "stale", 1);
        assert!(matches!(
            store.save(&stale).await,
            Err(Error::SessionConflict {
                stored_revision: 2,
                provided_revision: 1,
                ..
            })
        ));
        store.delete(&id).await.unwrap();
    }

    #[tokio::test]
    async fn initializes_schema_metadata() {
        let Some(url) = test_url() else {
            return;
        };
        let store = PostgresSessionStore::connect(&url).await.unwrap();
        assert_eq!(store.schema_version().await.unwrap(), STORE_SCHEMA_VERSION);
    }
}
