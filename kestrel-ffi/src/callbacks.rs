use std::{
    ffi::{CString, c_char, c_void},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Condvar, Mutex},
};

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
