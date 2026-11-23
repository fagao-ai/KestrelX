use std::{
    collections::VecDeque,
    ffi::{CStr, CString, c_char, c_void},
    ptr,
    sync::{Barrier, Mutex, mpsc},
    time::Duration,
};

use kestrel::{AgentOutput, FinishReason, Message, ModelResponse, ToolCall, Usage};
use kestrel_ffi::{
    KestrelAgent, kestrel_agent_cancel, kestrel_agent_free, kestrel_agent_new,
    kestrel_agent_new_with_model, kestrel_agent_register_tool, kestrel_agent_reset,
    kestrel_agent_run, kestrel_agent_session_json, kestrel_agent_set_event_callback,
    kestrel_agent_start_run, kestrel_agent_todos_json, kestrel_run_free, kestrel_run_is_done,
    kestrel_run_next_event, kestrel_run_wait, kestrel_string_free, kestrel_string_new,
};
use serde_json::{Value, json};

struct ModelState {
    responses: Mutex<VecDeque<String>>,
    requests: Mutex<Vec<Value>>,
}

struct EventState {
    events: Mutex<Vec<Value>>,
}

struct BlockingState {
    started: Barrier,
    release: Barrier,
}

unsafe extern "C" fn model_callback(
    user_data: *mut c_void,
    request_json: *const c_char,
) -> *mut c_char {
    // SAFETY: the test passes a live ModelState for the duration of the agent.
    let state = unsafe { &*(user_data as *const ModelState) };
    // SAFETY: Kestrel passes a valid request string.
    let request = unsafe { CStr::from_ptr(request_json) }.to_str().unwrap();
    state
        .requests
        .lock()
        .unwrap()
        .push(serde_json::from_str(request).unwrap());
    let response = state
        .responses
        .lock()
        .unwrap()
        .pop_front()
        .expect("scripted model response");
    allocate(&response)
}

unsafe extern "C" fn event_callback(user_data: *mut c_void, event_json: *const c_char) {
    // SAFETY: the test passes a live EventState for the duration of the agent.
    let state = unsafe { &*(user_data as *const EventState) };
    // SAFETY: Kestrel passes a valid event string for this callback.
    let event = unsafe { CStr::from_ptr(event_json) }.to_str().unwrap();
    state
        .events
        .lock()
        .unwrap()
        .push(serde_json::from_str(event).unwrap());
}

unsafe extern "C" fn blocking_model_callback(
    user_data: *mut c_void,
    _request_json: *const c_char,
) -> *mut c_char {
    // SAFETY: the test keeps the state alive until the callback completes.
    let state = unsafe { &*(user_data as *const BlockingState) };
    state.started.wait();
    state.release.wait();
    allocate(
        &serde_json::to_string(&ModelResponse {
            message: Message::assistant("late response"),
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
            model: Some("blocking".to_owned()),
            raw: None,
        })
        .unwrap(),
    )
}

unsafe extern "C" fn echo_tool(
    _user_data: *mut c_void,
    _call_id: *const c_char,
    arguments_json: *const c_char,
) -> *mut c_char {
    // SAFETY: Kestrel passes a valid arguments string.
    let arguments: Value =
        serde_json::from_str(unsafe { CStr::from_ptr(arguments_json) }.to_str().unwrap()).unwrap();
    allocate(
        &json!({
            "content": format!("echo:{}", arguments["text"].as_str().unwrap_or_default()),
            "is_error": false
        })
        .to_string(),
    )
}

fn allocate(value: &str) -> *mut c_char {
    let value = CString::new(value).unwrap();
    // SAFETY: value is a valid C string.
    unsafe { kestrel_string_new(value.as_ptr()) }
}

#[test]
fn callback_model_tool_events_and_session_round_trip() {
    let first = ModelResponse {
        message: Message::assistant_with_tool_calls(
            "",
            vec![ToolCall {
                id: "call-1".to_owned(),
                name: "echo".to_owned(),
                arguments: json!({ "text": "hello" }),
            }],
        ),
        finish_reason: FinishReason::ToolCalls,
        usage: Usage {
            input_tokens: 5,
            output_tokens: 2,
        },
        model: Some("callback".to_owned()),
        raw: None,
    };
    let second = ModelResponse {
        message: Message::assistant("tool complete"),
        finish_reason: FinishReason::Stop,
        usage: Usage {
            input_tokens: 7,
            output_tokens: 3,
        },
        model: Some("callback".to_owned()),
        raw: None,
    };
    let model_state = Box::new(ModelState {
        responses: Mutex::new(VecDeque::from([
            serde_json::to_string(&first).unwrap(),
            serde_json::to_string(&second).unwrap(),
        ])),
        requests: Mutex::new(Vec::new()),
    });
    let event_state = Box::new(EventState {
        events: Mutex::new(Vec::new()),
    });
    let config = CString::new(
        json!({
            "max_turns": 4,
            "streaming": true,
            "todo": true,
            "openai": { "model": "callback" }
        })
        .to_string(),
    )
    .unwrap();
    let mut error = ptr::null_mut();

    // SAFETY: all pointers and callback state remain valid through the run.
    let agent = unsafe {
        kestrel_agent_new_with_model(
            config.as_ptr(),
            Some(model_callback),
            (&*model_state as *const ModelState).cast_mut().cast(),
            &mut error,
        )
    };
    assert!(!agent.is_null(), "{}", error_message(error));
    // SAFETY: agent is valid and states live until after it is freed.
    assert_eq!(
        unsafe {
            kestrel_agent_set_event_callback(
                agent,
                Some(event_callback),
                (&*event_state as *const EventState).cast_mut().cast(),
            )
        },
        0
    );

    let tool_name = CString::new("echo").unwrap();
    let tool_description = CString::new("Echo text").unwrap();
    let schema = CString::new(
        json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        })
        .to_string(),
    )
    .unwrap();
    // SAFETY: all pointers are valid for this call.
    let status = unsafe {
        kestrel_agent_register_tool(
            agent,
            tool_name.as_ptr(),
            tool_description.as_ptr(),
            schema.as_ptr(),
            Some(echo_tool),
            ptr::null_mut(),
            &mut error,
        )
    };
    assert_eq!(status, 0, "{}", error_message(error));

    let prompt = CString::new("use the echo tool").unwrap();
    // SAFETY: agent and prompt are valid.
    let output_ptr = unsafe { kestrel_agent_run(agent, prompt.as_ptr(), &mut error) };
    assert!(!output_ptr.is_null(), "{}", error_message(error));
    // SAFETY: output is allocated by Kestrel.
    let output_json = unsafe { CStr::from_ptr(output_ptr) }
        .to_str()
        .unwrap()
        .to_owned();
    // SAFETY: ownership is released once.
    unsafe { kestrel_string_free(output_ptr) };
    let output: AgentOutput = serde_json::from_str(&output_json).unwrap();
    assert_eq!(output.text, "tool complete");
    assert_eq!(output.turns, 2);

    let requests = model_state.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let tool_result = requests[1]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "tool")
        .unwrap();
    assert_eq!(tool_result["content"][0]["text"], "echo:hello",);

    let events = event_state.events.lock().unwrap();
    assert!(events.iter().any(|event| event["type"] == "tool_completed"));
    assert!(events.iter().any(|event| {
        event["type"] == "model_delta"
            && event["delta"]["type"] == "text"
            && event["delta"]["text"] == "tool complete"
    }));
    assert!(events.iter().any(|event| event["type"] == "run_completed"));

    // SAFETY: agent is valid.
    let todos_ptr = unsafe { kestrel_agent_todos_json(agent, &mut error) };
    // SAFETY: todos JSON is allocated by Kestrel.
    let todos_json = unsafe { CStr::from_ptr(todos_ptr) }.to_str().unwrap();
    assert_eq!(todos_json, "[]");
    // SAFETY: ownership is released once.
    unsafe { kestrel_string_free(todos_ptr) };

    // SAFETY: agent is valid.
    let session_ptr = unsafe { kestrel_agent_session_json(agent, &mut error) };
    // SAFETY: session JSON is allocated by Kestrel.
    let session_json = unsafe { CStr::from_ptr(session_ptr) }.to_str().unwrap();
    let session: Value = serde_json::from_str(session_json).unwrap();
    assert_eq!(session["messages"].as_array().unwrap().len(), 4);
    // SAFETY: ownership is released once.
    unsafe { kestrel_string_free(session_ptr) };

    // SAFETY: no run is active.
    assert_eq!(unsafe { kestrel_agent_reset(agent, &mut error) }, 0);
    // SAFETY: agent ownership is transferred exactly once.
    unsafe { kestrel_agent_free(agent) };
}

#[test]
fn invalid_openai_config_returns_error_string() {
    let config = CString::new("{}").unwrap();
    let mut error = ptr::null_mut();
    // SAFETY: pointers are valid for this call.
    let agent = unsafe { kestrel_agent_new(config.as_ptr(), &mut error) };
    assert!(agent.is_null());
    assert!(error_message(error).contains("openai.model"));
    // SAFETY: error string is owned by the caller.
    unsafe { kestrel_string_free(error) };
}

#[test]
fn cancellation_waits_for_in_flight_host_callbacks() {
    let state = Box::new(BlockingState {
        started: Barrier::new(2),
        release: Barrier::new(2),
    });
    let config = CString::new(json!({ "openai": { "model": "blocking" } }).to_string()).unwrap();
    let mut error = ptr::null_mut();
    // SAFETY: callback state lives until the run and agent are finished.
    let agent = unsafe {
        kestrel_agent_new_with_model(
            config.as_ptr(),
            Some(blocking_model_callback),
            (&*state as *const BlockingState).cast_mut().cast(),
            &mut error,
        )
    };
    assert!(!agent.is_null(), "{}", error_message(error));

    let (sender, receiver) = mpsc::channel();
    let agent_address = agent as usize;
    let prompt = CString::new("cancel me").unwrap();
    let worker = std::thread::spawn(move || {
        let mut error = ptr::null_mut();
        // SAFETY: the main thread keeps the agent alive until this thread joins.
        let output = unsafe {
            kestrel_agent_run(
                agent_address as *mut KestrelAgent,
                prompt.as_ptr(),
                &mut error,
            )
        };
        let message = if error.is_null() {
            String::new()
        } else {
            // SAFETY: out_error is owned by this call.
            let message = unsafe { CStr::from_ptr(error) }
                .to_str()
                .unwrap()
                .to_owned();
            // SAFETY: ownership is released once.
            unsafe { kestrel_string_free(error) };
            message
        };
        if !output.is_null() {
            // SAFETY: output is owned by this call.
            unsafe { kestrel_string_free(output) };
        }
        sender.send(message).unwrap();
    });

    state.started.wait();
    // SAFETY: agent is valid and the run is active.
    assert_eq!(unsafe { kestrel_agent_cancel(agent) }, 0);
    assert!(matches!(
        receiver.recv_timeout(Duration::from_millis(100)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    state.release.wait();
    let error = receiver.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(error.contains("cancelled"), "{error}");
    worker.join().unwrap();
    // SAFETY: ownership is transferred exactly once.
    unsafe { kestrel_agent_free(agent) };
}

#[test]
fn async_run_polls_events_and_waits_for_output() {
    let model_state = Box::new(ModelState {
        responses: Mutex::new(VecDeque::from([serde_json::to_string(&ModelResponse {
            message: Message::assistant("async complete"),
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
            model: Some("callback".to_owned()),
            raw: None,
        })
        .unwrap()])),
        requests: Mutex::new(Vec::new()),
    });
    let config = CString::new(
        json!({
            "streaming": true,
            "openai": { "model": "callback" }
        })
        .to_string(),
    )
    .unwrap();
    let mut error = ptr::null_mut();
    // SAFETY: state outlives the run and agent.
    let agent = unsafe {
        kestrel_agent_new_with_model(
            config.as_ptr(),
            Some(model_callback),
            (&*model_state as *const ModelState).cast_mut().cast(),
            &mut error,
        )
    };
    assert!(!agent.is_null(), "{}", error_message(error));

    let prompt = CString::new("run asynchronously").unwrap();
    // SAFETY: agent and prompt are valid.
    let run = unsafe { kestrel_agent_start_run(agent, prompt.as_ptr(), &mut error) };
    assert!(!run.is_null(), "{}", error_message(error));

    let mut events = Vec::new();
    loop {
        let mut event_ptr = ptr::null_mut();
        // SAFETY: run is valid and event_ptr is writable.
        let status = unsafe { kestrel_run_next_event(run, 5_000, &mut event_ptr, &mut error) };
        match status {
            1 => {
                // SAFETY: event_ptr is allocated by Kestrel.
                let event = unsafe { CStr::from_ptr(event_ptr) }
                    .to_str()
                    .unwrap()
                    .to_owned();
                // SAFETY: ownership is released once.
                unsafe { kestrel_string_free(event_ptr) };
                events.push(serde_json::from_str::<Value>(&event).unwrap());
            }
            0 => continue,
            2 => break,
            -1 => panic!("{}", error_message(error)),
            other => panic!("unexpected event status {other}"),
        }
    }
    assert!(events.iter().any(|event| event["type"] == "run_completed"));

    // SAFETY: run is valid and the output pointer is writable.
    let mut output_ptr = ptr::null_mut();
    let status = unsafe { kestrel_run_wait(run, 5_000, &mut output_ptr, &mut error) };
    assert_eq!(status, 1, "{}", error_message(error));
    // SAFETY: output_ptr is allocated by Kestrel.
    let output = unsafe { CStr::from_ptr(output_ptr) }.to_str().unwrap();
    let output: AgentOutput = serde_json::from_str(output).unwrap();
    assert_eq!(output.text, "async complete");
    // SAFETY: ownership is released once.
    unsafe { kestrel_string_free(output_ptr) };
    // SAFETY: run is valid.
    assert_eq!(unsafe { kestrel_run_is_done(run) }, 1);
    // SAFETY: run is complete.
    unsafe { kestrel_run_free(run) };
    // SAFETY: no run is active.
    unsafe { kestrel_agent_free(agent) };
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
