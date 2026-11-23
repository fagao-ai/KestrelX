use std::{
    ffi::{CStr, CString, c_char, c_int, c_void},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Condvar, Mutex},
};

use kestrel::ModelDelta;

/// Tracks host callbacks that may still be running after a cancelled run.
#[derive(Default)]
pub(crate) struct CallbackTracker {
    active: Mutex<usize>,
    idle: Condvar,
}

impl CallbackTracker {
    pub(crate) fn enter(self: &Arc<Self>) -> ActiveCallback {
        let mut active = self
            .active
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *active += 1;
        drop(active);
        ActiveCallback {
            tracker: self.clone(),
        }
    }

    pub(crate) fn wait_idle(&self) {
        let mut active = self
            .active
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        while *active > 0 {
            active = self
                .idle
                .wait(active)
                .unwrap_or_else(|error| error.into_inner());
        }
    }
}

pub(crate) struct ActiveCallback {
    tracker: Arc<CallbackTracker>,
}

impl Drop for ActiveCallback {
    fn drop(&mut self) {
        let mut active = self
            .tracker
            .active
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *active = active.saturating_sub(1);
        if *active == 0 {
            self.tracker.idle.notify_all();
        }
    }
}

pub type EventCallbackFn = unsafe extern "C" fn(user_data: *mut c_void, event_json: *const c_char);
pub type ToolCallbackFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    call_id: *const c_char,
    arguments_json: *const c_char,
) -> *mut c_char;
pub type ModelCallbackFn =
    unsafe extern "C" fn(user_data: *mut c_void, request_json: *const c_char) -> *mut c_char;
pub type DeltaEmitterFn = unsafe extern "C" fn(emit_ctx: *mut c_void, delta_json: *const c_char);
pub type StreamingModelCallbackFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    request_json: *const c_char,
    emit: DeltaEmitterFn,
    emit_ctx: *mut c_void,
    error: *mut *mut c_char,
) -> *mut c_char;
pub type SessionLoadCallbackFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    session_id: *const c_char,
    error: *mut *mut c_char,
) -> *mut c_char;
pub type SessionSaveCallbackFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    session_json: *const c_char,
    error: *mut *mut c_char,
) -> c_int;
pub type SessionDeleteCallbackFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    session_id: *const c_char,
    error: *mut *mut c_char,
) -> c_int;
pub type MemoryRecallCallbackFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    query_json: *const c_char,
    error: *mut *mut c_char,
) -> *mut c_char;
pub type MemoryRememberCallbackFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    entry_json: *const c_char,
    error: *mut *mut c_char,
) -> c_int;

#[derive(Clone, Copy)]
pub(crate) struct EventCallback {
    pub function: EventCallbackFn,
    pub user_data: usize,
}

impl EventCallback {
    pub(crate) fn invoke(self, event_json: &str) -> Result<(), String> {
        let event = CString::new(event_json)
            .map_err(|error| format!("event JSON contains NUL: {error}"))?;
        catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: the host promises that `user_data` is valid and that the
            // callback follows the C ABI. The string is valid for this call.
            unsafe {
                (self.function)(self.user_data as *mut c_void, event.as_ptr());
            }
        }))
        .map_err(|_| "event callback panicked".to_owned())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ToolCallback {
    pub function: ToolCallbackFn,
    pub user_data: usize,
}

impl ToolCallback {
    pub(crate) fn invoke(self, call_id: &str, arguments_json: &str) -> Result<String, String> {
        let call_id =
            CString::new(call_id).map_err(|error| format!("call id contains NUL: {error}"))?;
        let arguments = CString::new(arguments_json)
            .map_err(|error| format!("tool arguments contain NUL: {error}"))?;
        let pointer = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: callback and user_data validity are part of the FFI
            // contract. All input strings outlive this call.
            unsafe {
                (self.function)(
                    self.user_data as *mut c_void,
                    call_id.as_ptr(),
                    arguments.as_ptr(),
                )
            }
        }))
        .map_err(|_| "tool callback panicked".to_owned())?;
        take_callback_string(pointer, "tool")
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ModelCallback {
    pub function: ModelCallbackFn,
    pub user_data: usize,
}

impl ModelCallback {
    pub(crate) fn invoke(self, request_json: &str) -> Result<String, String> {
        let request = CString::new(request_json)
            .map_err(|error| format!("model request contains NUL: {error}"))?;
        let pointer = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: callback and user_data validity are part of the FFI
            // contract. The request string outlives this call.
            unsafe { (self.function)(self.user_data as *mut c_void, request.as_ptr()) }
        }))
        .map_err(|_| "model callback panicked".to_owned())?;
        take_callback_string(pointer, "model")
    }
}

#[derive(Clone, Copy)]
pub(crate) struct StreamingModelCallback {
    pub function: StreamingModelCallbackFn,
    pub user_data: usize,
}

struct DeltaContext {
    sender: tokio::sync::mpsc::UnboundedSender<ModelDelta>,
}

unsafe extern "C" fn emit_model_delta(emit_ctx: *mut c_void, delta_json: *const c_char) {
    if emit_ctx.is_null() || delta_json.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: emit_ctx is a DeltaContext borrowed for this callback.
        let context = unsafe { &*(emit_ctx as *const DeltaContext) };
        // SAFETY: delta_json is a valid string for this callback.
        let delta_json = unsafe { CStr::from_ptr(delta_json) };
        if let Ok(text) = delta_json.to_str()
            && let Ok(delta) = serde_json::from_str::<ModelDelta>(text)
        {
            let _ = context.sender.send(delta);
        }
    }));
}

impl StreamingModelCallback {
    pub(crate) fn invoke(
        self,
        request_json: &str,
        sender: tokio::sync::mpsc::UnboundedSender<ModelDelta>,
    ) -> Result<String, String> {
        let request = CString::new(request_json)
            .map_err(|error| format!("model request contains NUL: {error}"))?;
        let context = DeltaContext { sender };
        let mut error = std::ptr::null_mut();
        let pointer = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: callback and user_data are provided by the host. The
            // emitter and context are borrowed only for this call.
            unsafe {
                (self.function)(
                    self.user_data as *mut c_void,
                    request.as_ptr(),
                    emit_model_delta,
                    (&context as *const DeltaContext).cast_mut().cast(),
                    &mut error,
                )
            }
        }))
        .map_err(|_| "streaming model callback panicked".to_owned())?;
        match callback_result(pointer, error, "streaming model")? {
            Some(value) => Ok(value),
            None => Err("streaming model callback returned NULL".to_owned()),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SessionLoadCallback {
    pub function: SessionLoadCallbackFn,
    pub user_data: usize,
}

impl SessionLoadCallback {
    pub(crate) fn invoke(self, session_id: &str) -> Result<Option<String>, String> {
        let session_id = CString::new(session_id)
            .map_err(|error| format!("session id contains NUL: {error}"))?;
        let mut error = std::ptr::null_mut();
        let pointer = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: callback and user_data are provided by the host.
            unsafe {
                (self.function)(
                    self.user_data as *mut c_void,
                    session_id.as_ptr(),
                    &mut error,
                )
            }
        }))
        .map_err(|_| "session load callback panicked".to_owned())?;
        callback_result(pointer, error, "session load")
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SessionSaveCallback {
    pub function: SessionSaveCallbackFn,
    pub user_data: usize,
}

impl SessionSaveCallback {
    pub(crate) fn invoke(self, session_json: &str) -> Result<(), String> {
        let session_json = CString::new(session_json)
            .map_err(|error| format!("session JSON contains NUL: {error}"))?;
        let mut error = std::ptr::null_mut();
        let status = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: callback and user_data are provided by the host.
            unsafe {
                (self.function)(
                    self.user_data as *mut c_void,
                    session_json.as_ptr(),
                    &mut error,
                )
            }
        }))
        .map_err(|_| "session save callback panicked".to_owned())?;
        callback_status(status, error, "session save")
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SessionDeleteCallback {
    pub function: SessionDeleteCallbackFn,
    pub user_data: usize,
}

impl SessionDeleteCallback {
    pub(crate) fn invoke(self, session_id: &str) -> Result<(), String> {
        let session_id = CString::new(session_id)
            .map_err(|error| format!("session id contains NUL: {error}"))?;
        let mut error = std::ptr::null_mut();
        let status = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: callback and user_data are provided by the host.
            unsafe {
                (self.function)(
                    self.user_data as *mut c_void,
                    session_id.as_ptr(),
                    &mut error,
                )
            }
        }))
        .map_err(|_| "session delete callback panicked".to_owned())?;
        callback_status(status, error, "session delete")
    }
}

#[derive(Clone, Copy)]
pub(crate) struct MemoryRecallCallback {
    pub function: MemoryRecallCallbackFn,
    pub user_data: usize,
}

impl MemoryRecallCallback {
    pub(crate) fn invoke(self, query_json: &str) -> Result<String, String> {
        let query_json = CString::new(query_json)
            .map_err(|error| format!("memory query contains NUL: {error}"))?;
        let mut error = std::ptr::null_mut();
        let pointer = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: callback and user_data are provided by the host.
            unsafe {
                (self.function)(
                    self.user_data as *mut c_void,
                    query_json.as_ptr(),
                    &mut error,
                )
            }
        }))
        .map_err(|_| "memory recall callback panicked".to_owned())?;
        match callback_result(pointer, error, "memory recall")? {
            Some(value) => Ok(value),
            None => Err("memory recall callback returned NULL".to_owned()),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct MemoryRememberCallback {
    pub function: MemoryRememberCallbackFn,
    pub user_data: usize,
}

impl MemoryRememberCallback {
    pub(crate) fn invoke(self, entry_json: &str) -> Result<(), String> {
        let entry_json = CString::new(entry_json)
            .map_err(|error| format!("memory entry contains NUL: {error}"))?;
        let mut error = std::ptr::null_mut();
        let status = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: callback and user_data are provided by the host.
            unsafe {
                (self.function)(
                    self.user_data as *mut c_void,
                    entry_json.as_ptr(),
                    &mut error,
                )
            }
        }))
        .map_err(|_| "memory remember callback panicked".to_owned())?;
        callback_status(status, error, "memory remember")
    }
}

pub(crate) fn take_callback_string(
    pointer: *mut c_char,
    callback_kind: &str,
) -> Result<String, String> {
    if pointer.is_null() {
        return Err(format!("{callback_kind} callback returned NULL"));
    }
    // SAFETY: the callback contract requires strings allocated by
    // kestrel_string_new, so Rust's allocator owns this pointer.
    let string = unsafe { CString::from_raw(pointer) };
    string
        .into_string()
        .map_err(|error| format!("{callback_kind} callback returned invalid UTF-8: {error}"))
}

fn callback_result(
    pointer: *mut c_char,
    error: *mut c_char,
    callback_kind: &str,
) -> Result<Option<String>, String> {
    let error = take_optional_callback_string(error)?;
    if let Some(error) = error {
        if !pointer.is_null() {
            // SAFETY: callback results must be allocated by kestrel_string_new.
            drop(unsafe { CString::from_raw(pointer) });
        }
        return Err(error);
    }
    if pointer.is_null() {
        Ok(None)
    } else {
        take_callback_string(pointer, callback_kind).map(Some)
    }
}

fn callback_status(status: c_int, error: *mut c_char, callback_kind: &str) -> Result<(), String> {
    let error = take_optional_callback_string(error)?;
    if let Some(error) = error {
        Err(error)
    } else if status == 0 {
        Ok(())
    } else {
        Err(format!("{callback_kind} callback returned status {status}"))
    }
}

fn take_optional_callback_string(pointer: *mut c_char) -> Result<Option<String>, String> {
    if pointer.is_null() {
        return Ok(None);
    }
    // SAFETY: callback errors must be allocated by kestrel_string_new.
    let value = unsafe { CString::from_raw(pointer) };
    value
        .into_string()
        .map(Some)
        .map_err(|error| format!("callback error is not valid UTF-8: {error}"))
}
