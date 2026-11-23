use std::{future::Future, sync::OnceLock};

use tokio::{
    runtime::{Builder, Runtime},
    task::JoinHandle,
};

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

pub(crate) fn block_on<F>(future: F) -> Result<F::Output, String>
where
    F: Future,
{
    if tokio::runtime::Handle::try_current().is_ok() {
        return Err("blocking Kestrel FFI calls cannot run on a Tokio runtime thread".to_owned());
    }
    Ok(runtime()?.block_on(future))
}

pub(crate) fn spawn<F>(future: F) -> Result<JoinHandle<F::Output>, String>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    Ok(runtime()?.handle().spawn(future))
}

fn runtime() -> Result<&'static Runtime, String> {
    if let Some(runtime) = RUNTIME.get() {
        return Ok(runtime);
    }
    {
        let workers = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(2)
            .clamp(2, 8);
        let created = Builder::new_multi_thread()
            .worker_threads(workers)
            .enable_all()
            .thread_name("kestrel-ffi")
            .build()
            .map_err(|error| format!("failed to create Kestrel runtime: {error}"))?;
        let _ = RUNTIME.set(created);
        RUNTIME
            .get()
            .ok_or_else(|| "failed to initialize Kestrel runtime".to_owned())
    }
}
