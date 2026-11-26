# kestrel-store-sqlite

SQLite adapter for Kestrel's `SessionStore`.

```toml
[dependencies]
kestrel = { path = "." }
kestrel-store-sqlite = { path = "./kestrel-store-sqlite" }
```

```rust
use std::sync::Arc;

use kestrel::Agent;
use kestrel_store_sqlite::SqliteSessionStore;

let sessions = Arc::new(SqliteSessionStore::open("kestrel.sqlite")?);
let agent = Agent::builder(model)
    .session_store(sessions)
    .session_id("user-42")
    .build()?;
```

## Storage model

- Each session is stored as versioned JSON in `kestrel_sessions.payload`.
- Columns mirror `id`, `revision`, `updated_at_ms`, and `schema_version` for
  diagnostics and future indexing.
- The adapter manages its own schema with `PRAGMA user_version`.
- A newer database schema is rejected instead of being modified.
- `revision` is checked in a `BEGIN IMMEDIATE` transaction. Stale writes return
  `kestrel::Error::SessionConflict`.
- `open_in_memory()` is available for tests and ephemeral runs.

SQLite is compiled from source (`rusqlite`'s `bundled` feature), so consumers do
not need a system SQLite installation.
