use std::{
    collections::{HashMap, VecDeque},
    ffi::{CStr, CString, c_char, c_int, c_void},
    fs,
    io::{BufRead, BufReader},
    process::{Command, Stdio},
    ptr,
    sync::{Barrier, Mutex, mpsc},
    time::Duration,
};

use kestrel::{AgentOutput, FinishReason, Message, ModelResponse, ToolCall, Usage};
use kestrel_ffi::{
    DeltaEmitterFn, KestrelAgent, kestrel_agent_builder_build, kestrel_agent_builder_free,
    kestrel_agent_builder_new_with_model, kestrel_agent_builder_set_memory,
    kestrel_agent_builder_set_session_store, kestrel_agent_cancel, kestrel_agent_free,
    kestrel_agent_new, kestrel_agent_new_with_model, kestrel_agent_new_with_streaming_model,
    kestrel_agent_register_tool, kestrel_agent_reset, kestrel_agent_run,
    kestrel_agent_session_json, kestrel_agent_set_event_callback, kestrel_agent_start_run,
    kestrel_agent_todos_json, kestrel_capabilities_json, kestrel_contract_version,
    kestrel_run_free, kestrel_run_is_done, kestrel_run_next_event, kestrel_run_wait,
    kestrel_string_free, kestrel_string_new, kestrel_validate_config,
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

#[derive(Default)]
struct PersistenceState {
    sessions: Mutex<HashMap<String, String>>,
    memory: Mutex<Vec<Value>>,
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

unsafe extern "C" fn streaming_model_callback(
    _user_data: *mut c_void,
    _request_json: *const c_char,
    emit: DeltaEmitterFn,
    emit_ctx: *mut c_void,
    error: *mut *mut c_char,
) -> *mut c_char {
    unsafe { *error = ptr::null_mut() };
    for text in ["stream ", "works"] {
        let delta = CString::new(json!({ "type": "text", "text": text }).to_string()).unwrap();
        // SAFETY: emit and emit_ctx are valid for this callback.
        unsafe { emit(emit_ctx, delta.as_ptr()) };
    }
    allocate(
        &serde_json::to_string(&ModelResponse {
            message: Message::assistant("stream works"),
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
            model: Some("streaming-callback".to_owned()),
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

unsafe extern "C" fn memory_recall(
    user_data: *mut c_void,
    _query_json: *const c_char,
    error: *mut *mut c_char,
) -> *mut c_char {
    unsafe { *error = ptr::null_mut() };
    // SAFETY: test state outlives the callback.
    let state = unsafe { &*(user_data as *const PersistenceState) };
    let entries = state.memory.lock().unwrap().clone();
    allocate(&serde_json::to_string(&entries).unwrap())
}

unsafe extern "C" fn memory_remember(
    user_data: *mut c_void,
    entry_json: *const c_char,
    error: *mut *mut c_char,
) -> c_int {
    unsafe { *error = ptr::null_mut() };
    // SAFETY: test state outlives the callback.
    let state = unsafe { &*(user_data as *const PersistenceState) };
    // SAFETY: Kestrel passes valid JSON.
    let entry = unsafe { CStr::from_ptr(entry_json) }.to_str().unwrap();
    state
        .memory
        .lock()
        .unwrap()
        .push(serde_json::from_str(entry).unwrap());
    0
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

#[test]
fn streaming_model_callback_emits_token_deltas() {
    let event_state = Box::new(EventState {
        events: Mutex::new(Vec::new()),
    });
    let config = CString::new(
        json!({
            "streaming": true,
            "openai": { "model": "streaming-callback" }
        })
        .to_string(),
    )
    .unwrap();
    let mut error = ptr::null_mut();
    // SAFETY: event state outlives the agent.
    let agent = unsafe {
        kestrel_agent_new_with_streaming_model(
            config.as_ptr(),
            Some(streaming_model_callback),
            ptr::null_mut(),
            &mut error,
        )
    };
    assert!(!agent.is_null(), "{}", error_message(error));
    // SAFETY: event state outlives the agent.
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
    let prompt = CString::new("stream").unwrap();
    // SAFETY: agent and prompt are valid.
    let output = unsafe { kestrel_agent_run(agent, prompt.as_ptr(), &mut error) };
    assert!(!output.is_null(), "{}", error_message(error));
    // SAFETY: output is owned by this call.
    unsafe { kestrel_string_free(output) };

    let events = event_state.events.lock().unwrap();
    let deltas = events
        .iter()
        .filter(|event| event["type"] == "model_delta")
        .map(|event| event["delta"]["text"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(deltas, ["stream ", "works"]);
    // SAFETY: no run is active.
    unsafe { kestrel_agent_free(agent) };
}

#[test]
fn loads_skills_and_injects_catalog() {
    let root = tempfile::tempdir().unwrap();
    let skill_dir = root.path().join("writer");
    fs::create_dir(&skill_dir).unwrap();
    fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: writer\ndescription: Write concise copy\n---\nUse short sentences.",
    )
    .unwrap();
    let model_state = Box::new(ModelState {
        responses: Mutex::new(VecDeque::from([serde_json::to_string(&ModelResponse {
            message: Message::assistant("skills ready"),
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
            "openai": { "model": "callback" },
            "skills": {
                "paths": [root.path().to_string_lossy()]
            }
        })
        .to_string(),
    )
    .unwrap();
    let mut error = ptr::null_mut();
    // SAFETY: state and root outlive the agent.
    let agent = unsafe {
        kestrel_agent_new_with_model(
            config.as_ptr(),
            Some(model_callback),
            (&*model_state as *const ModelState).cast_mut().cast(),
            &mut error,
        )
    };
    assert!(!agent.is_null(), "{}", error_message(error));
    let prompt = CString::new("write").unwrap();
    // SAFETY: agent and prompt are valid.
    let output = unsafe { kestrel_agent_run(agent, prompt.as_ptr(), &mut error) };
    assert!(!output.is_null(), "{}", error_message(error));
    // SAFETY: output is owned by this call.
    unsafe { kestrel_string_free(output) };

    let requests = model_state.requests.lock().unwrap();
    let request = &requests[0];
    assert!(
        request["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == "read_skill")
    );
    assert!(
        request["messages"][0]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("Available skills")
    );
    // SAFETY: no run is active.
    unsafe { kestrel_agent_free(agent) };
}

#[test]
fn connects_stdio_mcp_server_and_registers_tools() {
    if Command::new("python3").arg("--version").output().is_err() {
        return;
    }
    let server = r#"
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request.get("method")
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "mock", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "echo", "description": "Echo via MCP", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}]}
    elif method == "tools/call":
        args = request.get("params", {}).get("arguments", {})
        result = {"content": [{"type": "text", "text": "mcp:" + str(args.get("text", ""))}], "isError": False}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"#;
    let model_state = Box::new(ModelState {
        responses: Mutex::new(VecDeque::from([
            serde_json::to_string(&ModelResponse {
                message: Message::assistant_with_tool_calls(
                    "",
                    vec![ToolCall {
                        id: "mcp-call".to_owned(),
                        name: "echo".to_owned(),
                        arguments: json!({ "text": "hello" }),
                    }],
                ),
                finish_reason: FinishReason::ToolCalls,
                usage: Usage::default(),
                model: Some("callback".to_owned()),
                raw: None,
            })
            .unwrap(),
            serde_json::to_string(&ModelResponse {
                message: Message::assistant("mcp complete"),
                finish_reason: FinishReason::Stop,
                usage: Usage::default(),
                model: Some("callback".to_owned()),
                raw: None,
            })
            .unwrap(),
        ])),
        requests: Mutex::new(Vec::new()),
    });
    let config = CString::new(
        json!({
            "openai": { "model": "callback" },
            "mcp_servers": [{
                "command": "python3",
                "args": ["-u", "-c", server]
            }]
        })
        .to_string(),
    )
    .unwrap();
    let mut error = ptr::null_mut();
    // SAFETY: model state outlives the agent.
    let agent = unsafe {
        kestrel_agent_new_with_model(
            config.as_ptr(),
            Some(model_callback),
            (&*model_state as *const ModelState).cast_mut().cast(),
            &mut error,
        )
    };
    assert!(!agent.is_null(), "{}", error_message(error));
    let prompt = CString::new("use mcp").unwrap();
    // SAFETY: agent and prompt are valid.
    let output = unsafe { kestrel_agent_run(agent, prompt.as_ptr(), &mut error) };
    assert!(!output.is_null(), "{}", error_message(error));
    // SAFETY: output is owned by this call.
    unsafe { kestrel_string_free(output) };

    let requests = model_state.requests.lock().unwrap();
    let tool_result = requests[1]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "tool")
        .unwrap();
    assert_eq!(tool_result["content"][0]["text"], "mcp:hello");
    // SAFETY: no run is active; this also closes MCP child processes.
    unsafe { kestrel_agent_free(agent) };
}

#[test]
fn connects_streamable_http_mcp_server() {
    if Command::new("python3").arg("--version").output().is_err() {
        return;
    }
    let server = r#"
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *_args):
        pass
    def send_json(self, status, body=None, session=False):
        payload = b"" if body is None else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        if session:
            self.send_header("Mcp-Session-Id", "http-session-1")
        self.end_headers()
        self.wfile.write(payload)
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        method = request.get("method")
        if method == "initialize":
            self.send_json(200, {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "http-mock", "version": "1"},
                },
            }, True)
        elif "id" not in request:
            self.send_json(202)
        elif method == "tools/list":
            self.send_json(200, {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {"tools": [{
                    "name": "echo",
                    "description": "Echo via HTTP MCP",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                }]},
            })
        elif method == "tools/call":
            arguments = request.get("params", {}).get("arguments", {})
            response = json.dumps({
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "content": [{
                        "type": "text",
                        "text": "mcp-http:" + str(arguments.get("text", "")),
                    }],
                    "isError": False,
                },
            })
            payload = ("event: message\ndata: " + response + "\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_json(200, {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {},
            })
    def do_DELETE(self):
        self.send_json(204)

server = HTTPServer(("127.0.0.1", 0), Handler)
print(server.server_address[1], flush=True)
server.serve_forever()
"#;
    let mut child = Command::new("python3")
        .args(["-u", "-c", server])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut line = String::new();
    BufReader::new(stdout).read_line(&mut line).unwrap();
    let port = line.trim();
    assert!(!port.is_empty(), "mock HTTP MCP server did not start");

    let model_state = Box::new(ModelState {
        responses: Mutex::new(VecDeque::from([
            serde_json::to_string(&ModelResponse {
                message: Message::assistant_with_tool_calls(
                    "",
                    vec![ToolCall {
                        id: "http-call".to_owned(),
                        name: "echo".to_owned(),
                        arguments: json!({ "text": "hello" }),
                    }],
                ),
                finish_reason: FinishReason::ToolCalls,
                usage: Usage::default(),
                model: Some("callback".to_owned()),
                raw: None,
            })
            .unwrap(),
            serde_json::to_string(&ModelResponse {
                message: Message::assistant("http mcp complete"),
                finish_reason: FinishReason::Stop,
                usage: Usage::default(),
                model: Some("callback".to_owned()),
                raw: None,
            })
            .unwrap(),
        ])),
        requests: Mutex::new(Vec::new()),
    });
    let config = CString::new(
        json!({
            "openai": { "model": "callback" },
            "mcp_servers": [{
                "url": format!("http://127.0.0.1:{port}/mcp")
            }]
        })
        .to_string(),
    )
    .unwrap();
    let mut error = ptr::null_mut();
    // SAFETY: model state outlives the agent.
    let agent = unsafe {
        kestrel_agent_new_with_model(
            config.as_ptr(),
            Some(model_callback),
            (&*model_state as *const ModelState).cast_mut().cast(),
            &mut error,
        )
    };
    assert!(!agent.is_null(), "{}", error_message(error));
    let prompt = CString::new("use http mcp").unwrap();
    // SAFETY: agent and prompt are valid.
    let output = unsafe { kestrel_agent_run(agent, prompt.as_ptr(), &mut error) };
    assert!(!output.is_null(), "{}", error_message(error));
    // SAFETY: output is owned by this call.
    unsafe { kestrel_string_free(output) };

    let requests = model_state.requests.lock().unwrap();
    let tool_result = requests[1]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "tool")
        .unwrap();
    assert_eq!(tool_result["content"][0]["text"], "mcp-http:hello");
    // SAFETY: no run is active; this also terminates the HTTP session.
    unsafe { kestrel_agent_free(agent) };
    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn builder_persists_sessions_and_long_term_memory() {
    let model_state = Box::new(ModelState {
        responses: Mutex::new(VecDeque::from([
            serde_json::to_string(&ModelResponse {
                message: Message::assistant_with_tool_calls(
                    "",
                    vec![ToolCall {
                        id: "remember-1".to_owned(),
                        name: "memory_remember".to_owned(),
                        arguments: json!({ "text": "The user lives in Shanghai" }),
                    }],
                ),
                finish_reason: FinishReason::ToolCalls,
                usage: Usage::default(),
                model: Some("callback".to_owned()),
                raw: None,
            })
            .unwrap(),
            serde_json::to_string(&ModelResponse {
                message: Message::assistant("first answer"),
                finish_reason: FinishReason::Stop,
                usage: Usage::default(),
                model: Some("callback".to_owned()),
                raw: None,
            })
            .unwrap(),
            serde_json::to_string(&ModelResponse {
                message: Message::assistant("second answer"),
                finish_reason: FinishReason::Stop,
                usage: Usage::default(),
                model: Some("callback".to_owned()),
                raw: None,
            })
            .unwrap(),
        ])),
        requests: Mutex::new(Vec::new()),
    });
    let persistence = Box::new(PersistenceState::default());
    let config = CString::new(
        json!({
            "session_id": "persistent-session",
            "openai": { "model": "callback" },
            "memory": { "enabled": true, "tools": true }
        })
        .to_string(),
    )
    .unwrap();
    let mut error = ptr::null_mut();

    // SAFETY: callback state outlives both agents.
    let builder = unsafe {
        kestrel_agent_builder_new_with_model(
            config.as_ptr(),
            Some(model_callback),
            (&*model_state as *const ModelState).cast_mut().cast(),
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
                (&*persistence as *const PersistenceState).cast_mut().cast(),
                &mut error,
            )
        },
        0,
        "{}",
        error_message(error)
    );
    // SAFETY: builder and pointer are valid.
    assert_eq!(
        unsafe {
            kestrel_agent_builder_set_memory(
                builder,
                Some(memory_recall),
                Some(memory_remember),
                (&*persistence as *const PersistenceState).cast_mut().cast(),
                &mut error,
            )
        },
        0,
        "{}",
        error_message(error)
    );
    // SAFETY: builder is valid.
    let first_agent = unsafe { kestrel_agent_builder_build(builder, &mut error) };
    assert!(!first_agent.is_null(), "{}", error_message(error));
    // SAFETY: builder can be freed after build.
    unsafe { kestrel_agent_builder_free(builder) };
    let first_prompt = CString::new("first prompt").unwrap();
    // SAFETY: agent and prompt are valid.
    let first_output = unsafe { kestrel_agent_run(first_agent, first_prompt.as_ptr(), &mut error) };
    assert!(!first_output.is_null(), "{}", error_message(error));
    // SAFETY: output is owned by this call.
    unsafe { kestrel_string_free(first_output) };
    // SAFETY: no run is active.
    unsafe { kestrel_agent_free(first_agent) };

    // SAFETY: callback state remains valid.
    let builder = unsafe {
        kestrel_agent_builder_new_with_model(
            config.as_ptr(),
            Some(model_callback),
            (&*model_state as *const ModelState).cast_mut().cast(),
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
                (&*persistence as *const PersistenceState).cast_mut().cast(),
                &mut error,
            )
        },
        0
    );
    // SAFETY: builder and pointer are valid.
    assert_eq!(
        unsafe {
            kestrel_agent_builder_set_memory(
                builder,
                Some(memory_recall),
                Some(memory_remember),
                (&*persistence as *const PersistenceState).cast_mut().cast(),
                &mut error,
            )
        },
        0
    );
    // SAFETY: builder is valid.
    let second_agent = unsafe { kestrel_agent_builder_build(builder, &mut error) };
    assert!(!second_agent.is_null(), "{}", error_message(error));
    // SAFETY: builder can be freed after build.
    unsafe { kestrel_agent_builder_free(builder) };
    let second_prompt = CString::new("second prompt").unwrap();
    // SAFETY: agent and prompt are valid.
    let second_output =
        unsafe { kestrel_agent_run(second_agent, second_prompt.as_ptr(), &mut error) };
    assert!(!second_output.is_null(), "{}", error_message(error));
    // SAFETY: output is owned by this call.
    unsafe { kestrel_string_free(second_output) };

    let requests = model_state.requests.lock().unwrap();
    let text = serde_json::to_string(requests.last().unwrap()).unwrap();
    assert!(text.contains("first prompt"));
    assert!(text.contains("first answer"));
    assert!(text.contains("second prompt"));
    assert!(text.contains("Shanghai"));
    assert_eq!(persistence.memory.lock().unwrap().len(), 1);
    // SAFETY: no run is active.
    unsafe { kestrel_agent_free(second_agent) };
}

#[test]
fn exposes_versioned_contract_and_capabilities() {
    assert_eq!(kestrel_contract_version(), 1);
    let mut error = ptr::null_mut();
    // SAFETY: error pointer is valid.
    let capabilities_ptr = unsafe { kestrel_capabilities_json(&mut error) };
    assert!(!capabilities_ptr.is_null(), "{}", error_message(error));
    // SAFETY: capabilities is allocated by Kestrel.
    let capabilities: Value = serde_json::from_str(
        unsafe { CStr::from_ptr(capabilities_ptr) }
            .to_str()
            .unwrap(),
    )
    .unwrap();
    assert!(
        capabilities
            .as_array()
            .unwrap()
            .contains(&json!("mcp_http"))
    );
    // SAFETY: ownership is released once.
    unsafe { kestrel_string_free(capabilities_ptr) };
}

#[test]
fn validates_shared_full_fixture() {
    let fixture = CString::new(include_str!("../../kestrel-contract/fixtures/full.json")).unwrap();
    let mut error = ptr::null_mut();
    // SAFETY: fixture and error pointers are valid.
    let status = unsafe { kestrel_validate_config(fixture.as_ptr(), 1, &mut error) };
    assert_eq!(status, 0, "{}", error_message(error));
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
