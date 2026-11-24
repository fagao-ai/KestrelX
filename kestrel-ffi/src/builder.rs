use std::{
    ffi::{c_char, c_int, c_void},
    ptr,
    sync::Arc,
};

use kestrel::{Memory, SessionStore};

use crate::{
    agent::{KestrelAgent, build_agent},
    callbacks::{
        CallbackTracker, MemoryRecallCallback, MemoryRecallCallbackFn, MemoryRememberCallback,
        MemoryRememberCallbackFn, ModelCallback, ModelCallbackFn, SessionDeleteCallback,
        SessionDeleteCallbackFn, SessionLoadCallback, SessionLoadCallbackFn, SessionSaveCallback,
        SessionSaveCallbackFn, StreamingModelCallback, StreamingModelCallbackFn,
    },
    config::{Settings, SettingsExt, parse_settings},
    ffi_catch,
    persistence::{CallbackMemory, CallbackSessionStore},
    strings::{clear_error, required_string, set_error},
};

/// Opaque builder for agents that need host persistence services.
pub struct KestrelAgentBuilder {
    settings: Option<Settings>,
    model: Option<ModelSource>,
    tracker: Arc<CallbackTracker>,
    session_store: Option<Arc<dyn SessionStore>>,
    memory: Option<Arc<dyn Memory>>,
}

enum ModelSource {
    OpenAi,
    Callback(ModelCallback),
    StreamingCallback(StreamingModelCallback),
}

impl KestrelAgentBuilder {
    fn new(settings: Settings, model: ModelSource) -> Self {
        Self {
            settings: Some(settings),
            model: Some(model),
            tracker: Arc::new(CallbackTracker::default()),
            session_store: None,
            memory: None,
        }
    }
}

unsafe fn builder_mut<'a>(
    builder: *mut KestrelAgentBuilder,
) -> Result<&'a mut KestrelAgentBuilder, String> {
    // SAFETY: the caller promises exclusive access to a live builder handle.
    unsafe { builder.as_mut() }.ok_or_else(|| "agent builder must not be NULL".to_owned())
}

/// Creates a builder for an OpenAI-compatible agent.
///
/// # Safety
///
/// Pointers follow the C ABI contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_agent_builder_new(
    config_json: *const c_char,
    error: *mut *mut c_char,
) -> *mut KestrelAgentBuilder {
    // SAFETY: out_error follows the C contract.
    unsafe { clear_error(error) };
    let result = ffi_catch(|| {
        let config_json = unsafe { required_string(config_json, "config_json") }?;
        let settings = parse_settings(&config_json, false)?;
        Ok(KestrelAgentBuilder::new(settings, ModelSource::OpenAi))
    });
    match result {
        Ok(builder) => Box::into_raw(Box::new(builder)),
        Err(error_message) => {
            // SAFETY: out_error follows the C contract.
            unsafe { set_error(error, error_message) };
            ptr::null_mut()
        }
    }
}

/// Creates a builder with a host-language model callback.
///
/// # Safety
///
/// Callback and pointers follow the C ABI contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_agent_builder_new_with_model(
    config_json: *const c_char,
    callback: Option<ModelCallbackFn>,
    user_data: *mut c_void,
    error: *mut *mut c_char,
) -> *mut KestrelAgentBuilder {
    // SAFETY: out_error follows the C contract.
    unsafe { clear_error(error) };
    let result = ffi_catch(|| {
        let config_json = unsafe { required_string(config_json, "config_json") }?;
        let settings = parse_settings(&config_json, true)?;
        let callback = callback.ok_or_else(|| "model callback must not be NULL".to_owned())?;
        Ok(KestrelAgentBuilder::new(
            settings,
            ModelSource::Callback(ModelCallback {
                function: callback,
                user_data: user_data as usize,
            }),
        ))
    });
    match result {
        Ok(builder) => Box::into_raw(Box::new(builder)),
        Err(error_message) => {
            // SAFETY: out_error follows the C contract.
            unsafe { set_error(error, error_message) };
            ptr::null_mut()
        }
    }
}

/// Creates a builder with a token-streaming host model callback.
///
/// # Safety
///
/// Callback and pointers follow the C ABI contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_agent_builder_new_with_streaming_model(
    config_json: *const c_char,
    callback: Option<StreamingModelCallbackFn>,
    user_data: *mut c_void,
    error: *mut *mut c_char,
) -> *mut KestrelAgentBuilder {
    // SAFETY: out_error follows the C contract.
    unsafe { clear_error(error) };
    let result = ffi_catch(|| {
        let config_json = unsafe { required_string(config_json, "config_json") }?;
        let settings = parse_settings(&config_json, true)?;
        let callback = callback.ok_or_else(|| "model callback must not be NULL".to_owned())?;
        Ok(KestrelAgentBuilder::new(
            settings,
            ModelSource::StreamingCallback(StreamingModelCallback {
                function: callback,
                user_data: user_data as usize,
            }),
        ))
    });
    match result {
        Ok(builder) => Box::into_raw(Box::new(builder)),
        Err(error_message) => {
            // SAFETY: out_error follows the C contract.
            unsafe { set_error(error, error_message) };
            ptr::null_mut()
        }
    }
}

/// Installs host-backed session persistence.
///
/// # Safety
///
/// Callbacks and user data follow the C ABI contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_agent_builder_set_session_store(
    builder: *mut KestrelAgentBuilder,
    load: Option<SessionLoadCallbackFn>,
    save: Option<SessionSaveCallbackFn>,
    delete: Option<SessionDeleteCallbackFn>,
    user_data: *mut c_void,
    error: *mut *mut c_char,
) -> c_int {
    // SAFETY: out_error follows the C contract.
    unsafe { clear_error(error) };
    let result = ffi_catch(|| {
        let builder = unsafe { builder_mut(builder) }?;
        let load = load.ok_or_else(|| "session load callback must not be NULL".to_owned())?;
        let save = save.ok_or_else(|| "session save callback must not be NULL".to_owned())?;
        let user_data = user_data as usize;
        builder.session_store = Some(Arc::new(CallbackSessionStore::new(
            SessionLoadCallback {
                function: load,
                user_data,
            },
            SessionSaveCallback {
                function: save,
                user_data,
            },
            delete.map(|function| SessionDeleteCallback {
                function,
                user_data,
            }),
            builder.tracker.clone(),
        )));
        Ok(())
    });
    status_result(result, error)
}

/// Installs host-backed long-term memory.
///
/// # Safety
///
/// Callbacks and user data follow the C ABI contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_agent_builder_set_memory(
    builder: *mut KestrelAgentBuilder,
    recall: Option<MemoryRecallCallbackFn>,
    remember: Option<MemoryRememberCallbackFn>,
    user_data: *mut c_void,
    error: *mut *mut c_char,
) -> c_int {
    // SAFETY: out_error follows the C contract.
    unsafe { clear_error(error) };
    let result = ffi_catch(|| {
        let builder = unsafe { builder_mut(builder) }?;
        let recall = recall.ok_or_else(|| "memory recall callback must not be NULL".to_owned())?;
        let remember =
            remember.ok_or_else(|| "memory remember callback must not be NULL".to_owned())?;
        let user_data = user_data as usize;
        builder.memory = Some(Arc::new(CallbackMemory::new(
            MemoryRecallCallback {
                function: recall,
                user_data,
            },
            MemoryRememberCallback {
                function: remember,
                user_data,
            },
            builder.tracker.clone(),
        )));
        Ok(())
    });
    status_result(result, error)
}

/// Builds the agent. The builder remains owned by the caller and may be freed
/// after this call. A builder can only produce one agent.
///
/// # Safety
///
/// Pointers follow the C ABI contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_agent_builder_build(
    builder: *mut KestrelAgentBuilder,
    error: *mut *mut c_char,
) -> *mut KestrelAgent {
    // SAFETY: out_error follows the C contract.
    unsafe { clear_error(error) };
    let result = ffi_catch(|| {
        let builder = unsafe { builder_mut(builder) }?;
        let settings = builder
            .settings
            .take()
            .ok_or_else(|| "agent builder was already used".to_owned())?;
        let model_source = builder
            .model
            .take()
            .ok_or_else(|| "agent builder was already used".to_owned())?;
        let tracker = builder.tracker.clone();
        let model = match model_source {
            ModelSource::OpenAi => settings.openai_model()?,
            ModelSource::Callback(callback) => settings.callback_model(callback, tracker.clone()),
            ModelSource::StreamingCallback(callback) => {
                settings.streaming_callback_model(callback, tracker.clone())
            }
        };
        let agent = build_agent(
            settings,
            model,
            tracker,
            builder.session_store.take(),
            builder.memory.take(),
        )?;
        Ok(Box::into_raw(agent))
    });
    match result {
        Ok(agent) => agent,
        Err(error_message) => {
            // SAFETY: out_error follows the C contract.
            unsafe { set_error(error, error_message) };
            ptr::null_mut()
        }
    }
}

/// Frees a builder that has not produced an agent.
///
/// # Safety
///
/// `builder` must be NULL or an owned builder handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kestrel_agent_builder_free(builder: *mut KestrelAgentBuilder) {
    if builder.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        drop(unsafe { Box::from_raw(builder) });
    }));
}

fn status_result(result: Result<(), String>, error: *mut *mut c_char) -> c_int {
    match result {
        Ok(()) => 0,
        Err(error_message) => {
            // SAFETY: out_error follows the C contract.
            unsafe { set_error(error, error_message) };
            -1
        }
    }
}
