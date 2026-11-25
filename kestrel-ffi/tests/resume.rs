use std::{
    collections::{HashMap, VecDeque},
    ffi::{CStr, CString, c_char, c_int, c_void},
    ptr,
    sync::Mutex,
};

use kestrel::{FinishReason, Message, ModelResponse, RunCheckpoint, Session, Usage};
use kestrel_ffi::{
    KestrelAgent, kestrel_agent_builder_build, kestrel_agent_builder_free,
    kestrel_agent_builder_new_with_model, kestrel_agent_builder_set_session_store,
    kestrel_agent_free, kestrel_agent_resume, kestrel_agent_start_resume, kestrel_run_free,
    kestrel_run_wait, kestrel_string_free, kestrel_string_new,
};
use serde_json::{Value, json};

struct ModelState {
    responses: Mutex<VecDeque<String>>,
}

#[derive(Default)]
struct PersistenceState {
    sessions: Mutex<HashMap<String, String>>,
}

unsafe extern "C" fn model_callback(
    user_data: *mut c_void,
    _request_json: *const c_char,
) -> *mut c_char {
    // SAFETY: the test passes a live ModelState for the duration of the agent.
    let state = unsafe { &*(user_data as *const ModelState) };
    let response = state
        .responses
        .lock()
        .unwrap()
        .pop_front()
        .expect("scripted model response");
    allocate(&response)
}

unsafe extern "C" fn session_load(
    user_data: *mut c_void,
    session_id: *const c_char,
    error: *mut *mut c_char,
) -> *mut c_char {
    unsafe { *error = ptr::null_mut() };
    // SAFETY: test state outlives the callback.
    let state = unsafe { &*(user_data as *const PersistenceState) };
    // SAFETY: Kestrel passes a valid session id.
    let id = unsafe { CStr::from_ptr(session_id) }.to_str().unwrap();
    state
        .sessions
        .lock()
        .unwrap()
        .get(id)
        .map(|json| allocate(json))
        .unwrap_or(ptr::null_mut())
}

unsafe extern "C" fn session_save(
    user_data: *mut c_void,
    session_json: *const c_char,
    error: *mut *mut c_char,
) -> c_int {
    unsafe { *error = ptr::null_mut() };
    // SAFETY: test state outlives the callback.
    let state = unsafe { &*(user_data as *const PersistenceState) };
    // SAFETY: Kestrel passes valid JSON.
    let json = unsafe { CStr::from_ptr(session_json) }
        .to_str()
        .unwrap()
        .to_owned();
    let session: Value = serde_json::from_str(&json).unwrap();
    state
        .sessions
        .lock()
        .unwrap()
        .insert(session["id"].as_str().unwrap().to_owned(), json);
    0
}

unsafe extern "C" fn session_delete(
    user_data: *mut c_void,
    session_id: *const c_char,
    error: *mut *mut c_char,
) -> c_int {
    unsafe { *error = ptr::null_mut() };
    // SAFETY: test state outlives the callback.
    let state = unsafe { &*(user_data as *const PersistenceState) };
    // SAFETY: Kestrel passes a valid session id.
    let id = unsafe { CStr::from_ptr(session_id) }.to_str().unwrap();
    state.sessions.lock().unwrap().remove(id);
    0
}

fn allocate(value: &str) -> *mut c_char {
    let value = CString::new(value).unwrap();
    // SAFETY: value is a valid C string.
    unsafe { kestrel_string_new(value.as_ptr()) }
}

fn error_message(error: *mut c_char) -> String {
    if error.is_null() {
        return "no error".to_owned();
    }
    // SAFETY: error is a NUL-terminated string returned by Kestrel.
    unsafe { CStr::from_ptr(error) }
        .to_str()
        .unwrap_or("invalid error string")
        .to_owned()
}

fn seed_resume_session(persistence: &PersistenceState, session_id: &str, run_id: &str) {
    let mut session = Session::with_id(session_id);
    session.push(Message::user("finish the interrupted run"));
    session.active_run = Some(RunCheckpoint::model_call(run_id, 1, Usage::default()));
    session.touch();
    persistence.sessions.lock().unwrap().insert(
        session_id.to_owned(),
        serde_json::to_string(&session).unwrap(),
    );
}

fn build_persistent_agent(
    config: &CString,
    model_state: &ModelState,
    persistence: &PersistenceState,
) -> *mut KestrelAgent {
    let mut error = ptr::null_mut();
    // SAFETY: callback state outlives the agent.
    let builder = unsafe {
        kestrel_agent_builder_new_with_model(
            config.as_ptr(),
            Some(model_callback),
            (model_state as *const ModelState).cast_mut().cast(),
            &mut error,
        )
    };
    assert!(!builder.is_null(), "{}", error_message(error));
    // SAFETY: builder and pointer are valid.
    assert_eq!(
        unsafe {
            kestrel_agent_builder_set_session_store(
                builder,
                Some(session_load),
                Some(session_save),
                Some(session_delete),
                (persistence as *const PersistenceState).cast_mut().cast(),
                &mut error,
            )
        },
        0,
        "{}",
        error_message(error)
    );
    // SAFETY: builder is valid.
    let agent = unsafe { kestrel_agent_builder_build(builder, &mut error) };
    assert!(!agent.is_null(), "{}", error_message(error));
    // SAFETY: builder can be freed after build.
    unsafe { kestrel_agent_builder_free(builder) };
    agent
}

#[test]
fn resumes_persisted_checkpoints_sync_and_async() {
    let model_state = Box::new(ModelState {
        responses: Mutex::new(VecDeque::from([
            serde_json::to_string(&ModelResponse {
                message: Message::assistant("sync resumed"),
                finish_reason: FinishReason::Stop,
                usage: Usage::default(),
                model: Some("callback".to_owned()),
                raw: None,
            })
            .unwrap(),
            serde_json::to_string(&ModelResponse {
                message: Message::assistant("async resumed"),
                finish_reason: FinishReason::Stop,
                usage: Usage::default(),
                model: Some("callback".to_owned()),
                raw: None,
            })
            .unwrap(),
        ])),
    });
    let persistence = Box::new(PersistenceState::default());
    let config = CString::new(
        json!({
            "session_id": "ffi-resume",
            "openai": { "model": "callback" }
        })
        .to_string(),
    )
    .unwrap();

    seed_resume_session(&persistence, "ffi-resume", "ffi-sync-run");
    let agent = build_persistent_agent(&config, &model_state, &persistence);
    let mut error = ptr::null_mut();
    // SAFETY: agent and error pointer are valid.
    let output_ptr = unsafe { kestrel_agent_resume(agent, 0, &mut error) };
    assert!(!output_ptr.is_null(), "{}", error_message(error));
    // SAFETY: Kestrel allocated the output string.
    let output: Value =
        serde_json::from_str(unsafe { CStr::from_ptr(output_ptr) }.to_str().unwrap()).unwrap();
    assert_eq!(output["run_id"], "ffi-sync-run");
    assert_eq!(output["text"], "sync resumed");
    // SAFETY: output is released exactly once.
    unsafe { kestrel_string_free(output_ptr) };
    // SAFETY: no run is active.
    unsafe { kestrel_agent_free(agent) };

    seed_resume_session(&persistence, "ffi-resume", "ffi-async-run");
    let agent = build_persistent_agent(&config, &model_state, &persistence);
    // SAFETY: agent and error pointer are valid.
    let run = unsafe { kestrel_agent_start_resume(agent, 0, &mut error) };
    assert!(!run.is_null(), "{}", error_message(error));
    let mut output_ptr = ptr::null_mut();
    // SAFETY: run and output pointers are valid.
    let status = unsafe { kestrel_run_wait(run, 5_000, &mut output_ptr, &mut error) };
    assert_eq!(status, 1, "{}", error_message(error));
    // SAFETY: Kestrel allocated the output string.
    let output: Value =
        serde_json::from_str(unsafe { CStr::from_ptr(output_ptr) }.to_str().unwrap()).unwrap();
    assert_eq!(output["run_id"], "ffi-async-run");
    assert_eq!(output["text"], "async resumed");
    // SAFETY: owned strings and handles are released once.
    unsafe {
        kestrel_string_free(output_ptr);
        kestrel_run_free(run);
        kestrel_agent_free(agent);
    }

    let stored: Value = serde_json::from_str(
        persistence
            .sessions
            .lock()
            .unwrap()
            .get("ffi-resume")
            .unwrap(),
    )
    .unwrap();
    assert!(stored.get("active_run").is_none());
}
