//! C ABI for Kestrel.
//!
//! The ABI is intentionally JSON-first: configuration, model requests, tool
//! arguments, events, sessions, and outputs are JSON strings. Only handles,
//! callbacks, and C strings cross the FFI boundary.

#![deny(unsafe_op_in_unsafe_fn)]

mod agent;
mod builder;
mod callbacks;
mod config;
mod contract;
mod event;
mod model;
mod persistence;
mod run;
mod runtime;
mod strings;
mod tools;

pub use agent::{
    KestrelAgent, kestrel_agent_cancel, kestrel_agent_delete_session, kestrel_agent_free,
    kestrel_agent_new, kestrel_agent_new_with_model, kestrel_agent_new_with_streaming_model,
    kestrel_agent_register_tool, kestrel_agent_reset, kestrel_agent_resume, kestrel_agent_run,
    kestrel_agent_run_message, kestrel_agent_session_json, kestrel_agent_set_event_callback,
    kestrel_agent_todos_json,
};
pub use builder::{
    KestrelAgentBuilder, kestrel_agent_builder_build, kestrel_agent_builder_free,
    kestrel_agent_builder_new, kestrel_agent_builder_new_with_model,
    kestrel_agent_builder_new_with_streaming_model, kestrel_agent_builder_set_memory,
    kestrel_agent_builder_set_session_store,
};
pub use callbacks::{
    DeltaEmitterFn, EventCallbackFn, MemoryRecallCallbackFn, MemoryRememberCallbackFn,
    ModelCallbackFn, SessionDeleteCallbackFn, SessionLoadCallbackFn, SessionSaveCallbackFn,
    StreamingModelCallbackFn, ToolCallbackFn,
};
pub use contract::{kestrel_capabilities_json, kestrel_contract_version, kestrel_validate_config};
pub use run::{
    KestrelRun, kestrel_agent_start_resume, kestrel_agent_start_run,
    kestrel_agent_start_run_message, kestrel_run_cancel, kestrel_run_free, kestrel_run_is_done,
    kestrel_run_next_event, kestrel_run_wait,
};
pub use strings::{kestrel_string_free, kestrel_string_new};

use std::any::Any;

/// Returns the FFI crate version as a static C string.
#[unsafe(no_mangle)]
pub extern "C" fn kestrel_version() -> *const std::ffi::c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

pub(crate) fn ffi_catch<T>(
    operation: impl FnOnce() -> std::result::Result<T, String>,
) -> std::result::Result<T, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)) {
        Ok(result) => result,
        Err(payload) => Err(format!("kestrel FFI panic: {}", panic_message(&payload))),
    }
}

fn panic_message(payload: &Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic".to_owned()
    }
}
