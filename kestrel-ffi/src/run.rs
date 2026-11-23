use std::{
    ffi::{c_char, c_int},
    ptr,
    sync::{
        Arc, Condvar, Mutex,
        mpsc::{self, Receiver, RecvTimeoutError, TryRecvError},
    },
    time::{Duration, Instant},
};

use kestrel::{AgentOutput, Message, RunOptions};
use tokio_util::sync::CancellationToken;

use crate::{
    agent::{KestrelAgent, agent_ref},
    event::RunEventSink,
    ffi_catch, runtime,
    strings::{clear_error, into_c_string, required_string, set_error},
};

/// Opaque asynchronous run handle.
pub struct KestrelRun {
    inner: Arc<RunInner>,
}

impl Drop for KestrelRun {
    fn drop(&mut self) {
        self.inner.cancellation.cancel();
    }
}

struct RunInner {
    cancellation: CancellationToken,
    events: Mutex<Receiver<String>>,
    completion: Mutex<Option<Result<String, String>>>,
    completion_changed: Condvar,
}

impl RunInner {
    fn new(cancellation: CancellationToken, events: Receiver<String>) -> Self {
        Self {
            cancellation,
            events: Mutex::new(events),
            completion: Mutex::new(None),
            completion_changed: Condvar::new(),
        }
    }

    fn complete(&self, result: Result<String, String>) {
        let mut completion = self
            .completion
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if completion.is_none() {
            *completion = Some(result);
            self.completion_changed.notify_all();
        }
    }

    fn is_done(&self) -> bool {
        self.completion
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .is_some()
    }

    fn next_event(&self, timeout: Option<Duration>) -> EventPoll {
        let receiver = self
            .events
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        match timeout {
            None => match receiver.recv() {
                Ok(event) => EventPoll::Event(event),
                Err(_) => EventPoll::End,
            },
            Some(timeout) if timeout.is_zero() => match receiver.try_recv() {
                Ok(event) => EventPoll::Event(event),
                Err(TryRecvError::Empty) => EventPoll::Timeout,
                Err(TryRecvError::Disconnected) => EventPoll::End,
            },
            Some(timeout) => match receiver.recv_timeout(timeout) {
                Ok(event) => EventPoll::Event(event),
                Err(RecvTimeoutError::Timeout) => EventPoll::Timeout,
                Err(RecvTimeoutError::Disconnected) => EventPoll::End,
            },
        }
    }

    fn wait(&self, timeout: Option<Duration>) -> WaitPoll {
        let mut completion = self
            .completion
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(result) = completion.as_ref() {
            return WaitPoll::Complete(result.clone());
        }

        match timeout {
            None => {
                while completion.is_none() {
                    completion = self
                        .completion_changed
                        .wait(completion)
                        .unwrap_or_else(|error| error.into_inner());
                }
                match completion.as_ref() {
                    Some(result) => WaitPoll::Complete(result.clone()),
                    None => WaitPoll::Timeout,
                }
            }
            Some(timeout) => {
                let deadline = Instant::now() + timeout;
                while completion.is_none() {
                    let now = Instant::now();
                    if now >= deadline {
                        return WaitPoll::Timeout;
                    }
                    let (next, _) = self
                        .completion_changed
                        .wait_timeout(completion, deadline - now)
                        .unwrap_or_else(|error| error.into_inner());
                    completion = next;
                }
                match completion.as_ref() {
                    Some(result) => WaitPoll::Complete(result.clone()),
                    None => WaitPoll::Timeout,
                }
            }
        }
    }
}

enum EventPoll {
    Event(String),
    Timeout,
    End,
}

enum WaitPoll {
    Complete(Result<String, String>),
    Timeout,
}

fn start_run(agent: &KestrelAgent, message: Message) -> Result<KestrelRun, String> {
    let cancellation = CancellationToken::new();
    let lease = agent.begin_run(cancellation.clone())?;
    let (event_sender, event_receiver) = mpsc::channel();
    let event_sink = Arc::new(RunEventSink::new(agent.events.clone(), event_sender));
    let run = Arc::new(RunInner::new(cancellation.clone(), event_receiver));
    let task_run = run.clone();
    let inner_agent = agent.inner.clone();

    let handle = runtime::spawn(async move {
        let result = inner_agent
            .run_with(
                message,
                RunOptions::default()
                    .with_cancellation(cancellation)
                    .with_event_sink(event_sink),
            )
            .await;
        let completion = match result {
            Ok(output) => serialize_output(&output),
            Err(error) => Err(error.to_string()),
        };
        drop(lease);
        task_run.complete(completion);
    })?;

    let monitor_run = run.clone();
    if let Ok(monitor) = runtime::spawn(async move {
        if let Err(error) = handle.await {
            monitor_run.complete(Err(format!("agent run task failed: {error}")));
        }
    }) {
        drop(monitor);
    }

    Ok(KestrelRun { inner: run })
}

fn serialize_output(output: &AgentOutput) -> Result<String, String> {
    serde_json::to_string(output).map_err(|error| format!("failed to serialize output: {error}"))
}

fn duration_from_ms(value: c_int) -> Option<Duration> {
    (value >= 0).then(|| Duration::from_millis(value as u64))
}

unsafe fn run_ref<'a>(run: *mut KestrelRun) -> Result<&'a KestrelRun, String> {
    // SAFETY: the caller promises a live run handle.
    unsafe { run.as_ref() }.ok_or_else(|| "run must not be NULL".to_owned())
}

/// Starts a nonblocking run for a plain text prompt.
///
/// # Safety
///
/// Pointers follow the C ABI contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_agent_start_run(
    agent: *mut KestrelAgent,
    prompt: *const c_char,
    error: *mut *mut c_char,
) -> *mut KestrelRun {
    // SAFETY: out_error follows the C contract.
    unsafe { clear_error(error) };
    let result = ffi_catch(|| {
        // SAFETY: pointers follow the C contract.
        let agent = unsafe { agent_ref(agent) }?;
        let prompt = unsafe { required_string(prompt, "prompt") }?;
        start_run(agent, Message::user(prompt))
    });
    match result {
        Ok(run) => Box::into_raw(Box::new(run)),
        Err(error_message) => {
            // SAFETY: out_error follows the C contract.
            unsafe { set_error(error, error_message) };
            ptr::null_mut()
        }
    }
}

/// Starts a nonblocking run for a `Message` JSON input.
///
/// # Safety
///
/// Pointers follow the C ABI contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_agent_start_run_message(
    agent: *mut KestrelAgent,
    message_json: *const c_char,
    error: *mut *mut c_char,
) -> *mut KestrelRun {
    // SAFETY: out_error follows the C contract.
    unsafe { clear_error(error) };
    let result = ffi_catch(|| {
        // SAFETY: pointers follow the C contract.
        let agent = unsafe { agent_ref(agent) }?;
        let message_json = unsafe { required_string(message_json, "message_json") }?;
        let message: Message = serde_json::from_str(&message_json)
            .map_err(|error| format!("invalid message JSON: {error}"))?;
        start_run(agent, message)
    });
    match result {
        Ok(run) => Box::into_raw(Box::new(run)),
        Err(error_message) => {
            // SAFETY: out_error follows the C contract.
            unsafe { set_error(error, error_message) };
            ptr::null_mut()
        }
    }
}

/// Polls the next event.
///
/// Returns 1 with `event_json`, 0 on timeout, 2 when the run has ended, or -1
/// on an FFI error.
///
/// # Safety
///
/// Pointers follow the C ABI contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_run_next_event(
    run: *mut KestrelRun,
    timeout_ms: c_int,
    event_json: *mut *mut c_char,
    error: *mut *mut c_char,
) -> c_int {
    // SAFETY: out pointers follow the C contract.
    unsafe {
        clear_error(error);
        if !event_json.is_null() {
            *event_json = ptr::null_mut();
        }
    }
    if event_json.is_null() {
        unsafe { set_error(error, "event_json must not be NULL".to_owned()) };
        return -1;
    }
    let result = ffi_catch(|| {
        // SAFETY: pointer follows the C contract.
        let run = unsafe { run_ref(run) }?;
        Ok(run.inner.next_event(duration_from_ms(timeout_ms)))
    });
    match result {
        Ok(EventPoll::Event(json)) => {
            // SAFETY: event_json is non-NULL and writable.
            unsafe { *event_json = into_c_string(json) };
            1
        }
        Ok(EventPoll::Timeout) => 0,
        Ok(EventPoll::End) => 2,
        Err(error_message) => {
            // SAFETY: out_error follows the C contract.
            unsafe { set_error(error, error_message) };
            -1
        }
    }
}

/// Waits for run completion.
///
/// Returns 1 with `output_json`, 0 on timeout, or -1 on run/FFI error.
///
/// # Safety
///
/// Pointers follow the C ABI contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_run_wait(
    run: *mut KestrelRun,
    timeout_ms: c_int,
    output_json: *mut *mut c_char,
    error: *mut *mut c_char,
) -> c_int {
    // SAFETY: out pointers follow the C contract.
    unsafe {
        clear_error(error);
        if !output_json.is_null() {
            *output_json = ptr::null_mut();
        }
    }
    if output_json.is_null() {
        unsafe { set_error(error, "output_json must not be NULL".to_owned()) };
        return -1;
    }
    let result = ffi_catch(|| {
        // SAFETY: pointer follows the C contract.
        let run = unsafe { run_ref(run) }?;
        Ok(run.inner.wait(duration_from_ms(timeout_ms)))
    });
    match result {
        Ok(WaitPoll::Complete(Ok(json))) => {
            // SAFETY: output_json is non-NULL and writable.
            unsafe { *output_json = into_c_string(json) };
            1
        }
        Ok(WaitPoll::Complete(Err(error_message))) => {
            // SAFETY: out_error follows the C contract.
            unsafe { set_error(error, error_message) };
            -1
        }
        Ok(WaitPoll::Timeout) => 0,
        Err(error_message) => {
            // SAFETY: out_error follows the C contract.
            unsafe { set_error(error, error_message) };
            -1
        }
    }
}

/// Returns whether the run has produced its final outcome.
///
/// # Safety
///
/// `run` must be NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_run_is_done(run: *mut KestrelRun) -> c_int {
    // SAFETY: pointer follows the C contract.
    let Some(run) = (unsafe { run.as_ref() }) else {
        return -1;
    };
    if run.inner.is_done() { 1 } else { 0 }
}

/// Requests cancellation of this run.
///
/// # Safety
///
/// `run` must be NULL or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_run_cancel(run: *mut KestrelRun) -> c_int {
    // SAFETY: pointer follows the C contract.
    let Some(run) = (unsafe { run.as_ref() }) else {
        return -1;
    };
    run.inner.cancellation.cancel();
    0
}

/// Cancels and frees a run handle, waiting for in-flight host callbacks.
///
/// # Safety
///
/// `run` must be NULL or an owned handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_run_free(run: *mut KestrelRun) {
    if run.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: ownership is transferred exactly once.
        let run = unsafe { Box::from_raw(run) };
        run.inner.cancellation.cancel();
        let _ = run.inner.wait(None);
    }));
}
