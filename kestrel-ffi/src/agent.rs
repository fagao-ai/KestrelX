use std::{
    ffi::{c_char, c_int, c_void},
    ptr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use kestrel::mcp::McpClient;
use kestrel::{
    Agent, AgentOutput, Memory, Message, Model, RunOptions, Session, SessionStore, TodoList,
};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::{
    callbacks::{
        CallbackTracker, EventCallback, EventCallbackFn, ModelCallback, ModelCallbackFn,
        ToolCallback, ToolCallbackFn,
    },
    config::Settings,
    event::{CallbackEventSink, EventSlot},
    ffi_catch,
    runtime::block_on,
    strings::{clear_error, into_c_string, required_string, set_error},
    tools::CallbackTool,
};

/// Opaque FFI agent handle.
pub struct KestrelAgent {
    pub(crate) inner: Agent,
    pub(crate) events: EventSlot,
    pub(crate) todos: Option<TodoList>,
    pub(crate) mcp_clients: Vec<McpClient>,
    tracker: Arc<CallbackTracker>,
    running: Arc<AtomicBool>,
    cancellation: Arc<Mutex<Option<CancellationToken>>>,
}

impl KestrelAgent {
    pub(crate) fn begin_run(&self, token: CancellationToken) -> Result<RunLease, String> {
        self.running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| "agent already has an active run".to_owned())?;
        *self
            .cancellation
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(token);
        Ok(RunLease {
            running: self.running.clone(),
            cancellation: self.cancellation.clone(),
            tracker: self.tracker.clone(),
        })
    }

    fn run_message(&self, message: Message) -> Result<AgentOutput, String> {
        let token = CancellationToken::new();
        let _guard = self.begin_run(token.clone())?;
        block_on(
            self.inner
                .run_with(message, RunOptions::default().with_cancellation(token)),
        )?
        .map_err(|error| error.to_string())
    }
}

pub(crate) struct RunLease {
    running: Arc<AtomicBool>,
    cancellation: Arc<Mutex<Option<CancellationToken>>>,
    tracker: Arc<CallbackTracker>,
}

impl Drop for RunLease {
    fn drop(&mut self) {
        *self
            .cancellation
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = None;
        self.tracker.wait_idle();
        self.running.store(false, Ordering::Release);
    }
}

pub(crate) fn build_agent(
    settings: Settings,
    model: Arc<dyn Model>,
    tracker: Arc<CallbackTracker>,
    session_store: Option<Arc<dyn SessionStore>>,
    memory: Option<Arc<dyn Memory>>,
) -> Result<Box<KestrelAgent>, String> {
    let events = EventSlot::default();
    let sink = Arc::new(CallbackEventSink::new(events.clone()));
    let built = settings.build(model, sink, session_store, memory)?;
    Ok(Box::new(KestrelAgent {
        inner: built.agent,
        events,
        todos: built.todos,
        mcp_clients: built.mcp_clients,
        tracker,
        running: Arc::new(AtomicBool::new(false)),
        cancellation: Arc::new(Mutex::new(None)),
    }))
}

pub(crate) unsafe fn agent_ref<'a>(agent: *mut KestrelAgent) -> Result<&'a KestrelAgent, String> {
    // SAFETY: the caller promises a live handle for this call.
    unsafe { agent.as_ref() }.ok_or_else(|| "agent must not be NULL".to_owned())
}

/// Builds an OpenAI-compatible agent from config JSON.
///
/// # Safety
///
/// `config_json` must be a valid NUL-terminated string. `error`, when non-NULL,
/// must point to writable storage for one `char *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_agent_new(
    config_json: *const c_char,
    error: *mut *mut c_char,
) -> *mut KestrelAgent {
    // SAFETY: out_error follows the C contract.
    unsafe { clear_error(error) };
    let result = ffi_catch(|| {
        // SAFETY: config_json is required by the C contract.
        let config_json = unsafe { required_string(config_json, "config_json") }?;
        let settings = Settings::parse(&config_json)?;
        let model = settings.openai_model()?;
        build_agent(
            settings,
            model,
            Arc::new(CallbackTracker::default()),
            None,
            None,
        )
    });

    match result {
        Ok(agent) => Box::into_raw(agent),
        Err(error_message) => {
            // SAFETY: out_error follows the C contract.
            unsafe { set_error(error, error_message) };
            ptr::null_mut()
        }
    }
}

/// Builds an agent with a host-language model callback.
///
/// # Safety
///
/// Strings and callback/user data follow the same lifetime rules as
/// [`kestrel_agent_new`]. The callback must return memory allocated by
/// `kestrel_string_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_agent_new_with_model(
    config_json: *const c_char,
    callback: Option<ModelCallbackFn>,
    user_data: *mut c_void,
    error: *mut *mut c_char,
) -> *mut KestrelAgent {
    // SAFETY: out_error follows the C contract.
    unsafe { clear_error(error) };
    let result = ffi_catch(|| {
        // SAFETY: config_json is required by the C contract.
        let config_json = unsafe { required_string(config_json, "config_json") }?;
        let settings = Settings::parse(&config_json)?;
        let callback = callback.ok_or_else(|| "model callback must not be NULL".to_owned())?;
        let tracker = Arc::new(CallbackTracker::default());
        let model = settings.callback_model(
            ModelCallback {
                function: callback,
                user_data: user_data as usize,
            },
            tracker.clone(),
        );
        build_agent(settings, model, tracker, None, None)
    });

    match result {
        Ok(agent) => Box::into_raw(agent),
        Err(error_message) => {
            // SAFETY: out_error follows the C contract.
            unsafe { set_error(error, error_message) };
            ptr::null_mut()
        }
    }
}

/// Frees an agent. No run may be active.
///
/// # Safety
///
/// `agent` must be a handle returned by a Kestrel constructor and must not be
/// used after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_agent_free(agent: *mut KestrelAgent) {
    if agent.is_null() {
        return;
    }
    // SAFETY: ownership is transferred exactly once.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut agent = unsafe { Box::from_raw(agent) };
        let clients = std::mem::take(&mut agent.mcp_clients);
        if !clients.is_empty() {
            let _ = block_on(async move {
                for client in clients {
                    let _ = client.close().await;
                }
            });
        }
        drop(agent);
    }));
}

/// Installs or clears the event callback.
///
/// # Safety
///
/// Callback and user data must follow the C contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_agent_set_event_callback(
    agent: *mut KestrelAgent,
    callback: Option<EventCallbackFn>,
    user_data: *mut c_void,
) -> c_int {
    // SAFETY: the agent pointer follows the C contract.
    let Some(agent) = (unsafe { agent.as_ref() }) else {
        return -1;
    };
    agent.events.set(callback.map(|function| EventCallback {
        function,
        user_data: user_data as usize,
    }));
    0
}

/// Registers one synchronous host tool.
///
/// # Safety
///
/// All pointer arguments must follow the C contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_agent_register_tool(
    agent: *mut KestrelAgent,
    name: *const c_char,
    description: *const c_char,
    input_schema_json: *const c_char,
    callback: Option<ToolCallbackFn>,
    user_data: *mut c_void,
    error: *mut *mut c_char,
) -> c_int {
    // SAFETY: out_error follows the C contract.
    unsafe { clear_error(error) };
    let result = ffi_catch(|| {
        // SAFETY: the agent pointer follows the C contract.
        let agent = unsafe { agent_ref(agent) }?;
        // SAFETY: all strings are required by the C contract.
        let name = unsafe { required_string(name, "tool name") }?;
        let description = unsafe { required_string(description, "tool description") }?;
        let schema_json = unsafe { required_string(input_schema_json, "tool schema") }?;
        let schema: Value = serde_json::from_str(&schema_json)
            .map_err(|error| format!("invalid tool schema JSON: {error}"))?;
        if !schema.is_object() {
            return Err("tool schema must be a JSON object".to_owned());
        }
        let callback = callback.ok_or_else(|| "tool callback must not be NULL".to_owned())?;
        let tool = CallbackTool::new(
            kestrel::ToolSpec::new(name, description, schema),
            ToolCallback {
                function: callback,
                user_data: user_data as usize,
            },
            agent.tracker.clone(),
        );
        agent
            .inner
            .register_tool(Arc::new(tool))
            .map_err(|error| error.to_string())
    });

    match result {
        Ok(()) => 0,
        Err(error_message) => {
            // SAFETY: out_error follows the C contract.
            unsafe { set_error(error, error_message) };
            -1
        }
    }
}

/// Runs one text prompt synchronously.
///
/// # Safety
///
/// `agent` must be a live handle, `prompt` must be a valid C string, and
/// `error` must be a valid error output location.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_agent_run(
    agent: *mut KestrelAgent,
    prompt: *const c_char,
    error: *mut *mut c_char,
) -> *mut c_char {
    // SAFETY: out_error follows the C contract.
    unsafe { clear_error(error) };
    let result = ffi_catch(|| {
        // SAFETY: arguments follow the C contract.
        let agent = unsafe { agent_ref(agent) }?;
        let prompt = unsafe { required_string(prompt, "prompt") }?;
        let output = agent.run_message(Message::user(prompt))?;
        serde_json::to_string(&output)
            .map_err(|error| format!("failed to serialize output: {error}"))
    });

    match result {
        Ok(json) => into_c_string(json),
        Err(error_message) => {
            // SAFETY: out_error follows the C contract.
            unsafe { set_error(error, error_message) };
            ptr::null_mut()
        }
    }
}

/// Runs one `Message` JSON input synchronously.
///
/// # Safety
///
/// Same pointer requirements as [`kestrel_agent_run`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_agent_run_message(
    agent: *mut KestrelAgent,
    message_json: *const c_char,
    error: *mut *mut c_char,
) -> *mut c_char {
    // SAFETY: out_error follows the C contract.
    unsafe { clear_error(error) };
    let result = ffi_catch(|| {
        // SAFETY: arguments follow the C contract.
        let agent = unsafe { agent_ref(agent) }?;
        let message_json = unsafe { required_string(message_json, "message_json") }?;
        let message: Message = serde_json::from_str(&message_json)
            .map_err(|error| format!("invalid message JSON: {error}"))?;
        let output = agent.run_message(message)?;
        serde_json::to_string(&output)
            .map_err(|error| format!("failed to serialize output: {error}"))
    });

    match result {
        Ok(json) => into_c_string(json),
        Err(error_message) => {
            // SAFETY: out_error follows the C contract.
            unsafe { set_error(error, error_message) };
            ptr::null_mut()
        }
    }
}

/// Cancels the currently active run.
///
/// # Safety
///
/// `agent` must be a live handle or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_agent_cancel(agent: *mut KestrelAgent) -> c_int {
    // SAFETY: agent pointer follows the C contract.
    let Some(agent) = (unsafe { agent.as_ref() }) else {
        return -1;
    };
    if let Some(token) = agent
        .cancellation
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clone()
    {
        token.cancel();
    }
    0
}

/// Returns a session JSON snapshot.
///
/// # Safety
///
/// Pointers follow the C contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_agent_session_json(
    agent: *mut KestrelAgent,
    error: *mut *mut c_char,
) -> *mut c_char {
    // SAFETY: out_error follows the C contract.
    unsafe { clear_error(error) };
    let result = ffi_catch(|| {
        // SAFETY: arguments follow the C contract.
        let agent = unsafe { agent_ref(agent) }?;
        let session: Session = block_on(agent.inner.session())?;
        serde_json::to_string(&session)
            .map_err(|error| format!("failed to serialize session: {error}"))
    });

    match result {
        Ok(json) => into_c_string(json),
        Err(error_message) => {
            // SAFETY: out_error follows the C contract.
            unsafe { set_error(error, error_message) };
            ptr::null_mut()
        }
    }
}

/// Returns the current todo list JSON, or NULL when todos are disabled.
///
/// # Safety
///
/// Pointers follow the C contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_agent_todos_json(
    agent: *mut KestrelAgent,
    error: *mut *mut c_char,
) -> *mut c_char {
    // SAFETY: out_error follows the C contract.
    unsafe { clear_error(error) };
    let result = ffi_catch(|| {
        // SAFETY: arguments follow the C contract.
        let agent = unsafe { agent_ref(agent) }?;
        agent
            .todos
            .as_ref()
            .map(|todos| {
                serde_json::to_string(&todos.snapshot())
                    .map_err(|error| format!("failed to serialize todos: {error}"))
            })
            .transpose()
    });

    match result {
        Ok(Some(json)) => into_c_string(json),
        Ok(None) => ptr::null_mut(),
        Err(error_message) => {
            // SAFETY: out_error follows the C contract.
            unsafe { set_error(error, error_message) };
            ptr::null_mut()
        }
    }
}

/// Resets conversation history while keeping the session ID.
///
/// # Safety
///
/// Pointers follow the C contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_agent_reset(
    agent: *mut KestrelAgent,
    error: *mut *mut c_char,
) -> c_int {
    // SAFETY: out_error follows the C contract.
    unsafe { clear_error(error) };
    let result = ffi_catch(|| {
        // SAFETY: arguments follow the C contract.
        let agent = unsafe { agent_ref(agent) }?;
        let _guard = agent.begin_run(CancellationToken::new())?;
        block_on(agent.inner.reset())?.map_err(|error| error.to_string())
    });

    match result {
        Ok(()) => 0,
        Err(error_message) => {
            // SAFETY: out_error follows the C contract.
            unsafe { set_error(error, error_message) };
            -1
        }
    }
}
