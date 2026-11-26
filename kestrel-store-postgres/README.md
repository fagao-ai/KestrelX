# kestrel-store-postgres

PostgreSQL adapter for Kestrel's `SessionStore`.

```rust
use std::sync::Arc;

use kestrel::Agent;
use kestrel_store_postgres::PostgresSessionStore;

let sessions = Arc::new(
    PostgresSessionStore::connect("postgres://localhost/kestrel").await?,
);
let agent = Agent::builder(model)
    .session_store(sessions)
    .session_id("user-42")
    .build()?;
```

## Storage model

- Each session is stored as versioned JSON in `kestrel_sessions.payload`.
- The adapter owns its table schema and stores its version in
  `kestrel_store_meta`.
- Per-session advisory locks make revision checks and writes atomic across
  processes. Stale writes return `kestrel::Error::SessionConflict`.
- A newer database schema is rejected instead of being modified.

## Testing

Set `KESTREL_POSTGRES_TEST_URL` to run integration tests:

```bash
docker run --rm -d --name kestrel-postgres-test \
  -e POSTGRES_PASSWORD=postgres -p 55432:5432 postgres:16-alpine
KESTREL_POSTGRES_TEST_URL=postgres://postgres:postgres@127.0.0.1:55432/postgres \
  cargo test -p kestrel-store-postgres
docker stop kestrel-postgres-test
```
