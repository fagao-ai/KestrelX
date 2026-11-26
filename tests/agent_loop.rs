mod support;

use std::{sync::Arc, time::Duration};

use kestrel::{
    Agent, AgentEventKind, Error, InMemorySessionStore, Message, ResumeOptions, RunCheckpoint,
    RunOptions, SESSION_SCHEMA_VERSION, Session, SessionStore, ToolOutput, ToolSpec, Usage,
    event_channel, function_tool,
};
use serde_json::json;
use support::{
    BlockingModel, ScriptedModel, StreamingModel, call, object_schema, text_response, tool_response,
};
use tokio_util::sync::CancellationToken;

fn echo_tool() -> Arc<dyn kestrel::Tool> {
    function_tool(
        ToolSpec::new("echo", "Echo text", object_schema()),
        |arguments, _context| async move {
            Ok(ToolOutput::new(
                arguments["text"].as_str().unwrap_or_default(),
            ))
        },
    )
}

#[tokio::test]
async fn executes_tools_and_emits_events() {
    let model = Arc::new(ScriptedModel::new([
        tool_response(vec![call("call_1", "echo", json!({ "text": "hello" }))]),
        text_response("finished"),
    ]));
    let (sender, mut receiver) = event_channel();
    let agent = Agent::builder(model.clone())
        .tool(echo_tool())
        .event_sink(Arc::new(sender))
        .max_turns(4)
        .build()
        .unwrap();

    let output = agent.run("use the tool").await.unwrap();
    assert_eq!(output.text, "finished");
    assert_eq!(output.turns, 2);
    assert_eq!(output.usage.input_tokens, 22);
    assert_eq!(output.usage.output_tokens, 9);

    let requests = model.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[1].messages.last().unwrap().role,
        kestrel::Role::Tool
    );
    assert_eq!(requests[1].messages.last().unwrap().text(), "hello");

    let messages = agent.messages().await;
    assert_eq!(
        messages
            .iter()
            .map(|message| message.role)
            .collect::<Vec<_>>(),
        vec![
            kestrel::Role::User,
            kestrel::Role::Assistant,
            kestrel::Role::Tool,
            kestrel::Role::Assistant
        ]
    );

    let mut events = Vec::new();
    while let Ok(event) = receiver.try_recv() {
        events.push(event);
    }
    assert!(matches!(events[0].kind, AgentEventKind::RunStarted));
    assert!(events.iter().any(|event| matches!(
        &event.kind,
        AgentEventKind::ToolCompleted {
            name,
            output,
            ..
        } if name == "echo" && output.content == "hello"
    )));
    assert!(matches!(
        events.last().map(|event| &event.kind),
        Some(AgentEventKind::RunCompleted { turns: 2, .. })
    ));
    assert!(events.iter().any(|event| matches!(
        &event.kind,
        AgentEventKind::ModelDelta {
            delta: kestrel::ModelDelta::Text { text },
            ..
        } if text == "finished"
    )));
    assert_eq!(events[0].run_id, output.run_id);
    assert_eq!(events[0].sequence, 1);
}

#[tokio::test]
async fn forwards_incremental_model_output() {
    let model = StreamingModel::new(["one", " two", " three"], text_response("one two three"));
    let (sender, mut receiver) = event_channel();
    let agent = Agent::builder(model)
        .event_sink(Arc::new(sender))
        .build()
        .unwrap();

    let output = agent.run("stream").await.unwrap();
    assert_eq!(output.text, "one two three");

    let mut chunks = Vec::new();
    while let Ok(event) = receiver.try_recv() {
        if let AgentEventKind::ModelDelta {
            delta: kestrel::ModelDelta::Text { text },
            ..
        } = event.kind
        {
            chunks.push(text);
        }
    }
    assert_eq!(chunks, ["one", " two", " three"]);
}

#[tokio::test]
#[tracing_test::traced_test]
async fn emits_structured_tracing_events() {
    let model = ScriptedModel::new([text_response("traced")]);
    let agent = Agent::builder(model).build().unwrap();

    let output = agent.run("trace this").await.unwrap();
    assert_eq!(output.text, "traced");
    assert!(logs_contain("model call completed"));
    assert!(logs_contain("agent run completed"));
}

#[tokio::test]
async fn feeds_recoverable_tool_errors_back_to_the_model() {
    let failing = function_tool(
        ToolSpec::new("explode", "Always fails", object_schema()),
        |_arguments, _context| async move { Err(Error::tool("explode", "boom")) },
    );
    let model = Arc::new(ScriptedModel::new([
        tool_response(vec![call("call_1", "explode", json!({}))]),
        text_response("recovered"),
    ]));
    let agent = Agent::builder(model.clone()).tool(failing).build().unwrap();

    let output = agent.run("try").await.unwrap();
    assert_eq!(output.text, "recovered");
    let requests = model.requests();
    let result = requests[1].messages.last().unwrap();
    assert_eq!(result.text(), "Tool error: tool `explode` failed: boom");
}

#[tokio::test]
async fn can_fail_fast_on_tool_errors() {
    let failing = function_tool(
        ToolSpec::new("explode", "Always fails", object_schema()),
        |_arguments, _context| async move { Err(Error::tool("explode", "boom")) },
    );
    let model = Arc::new(ScriptedModel::new([tool_response(vec![call(
        "call_1",
        "explode",
        json!({}),
    )])]));
    let agent = Agent::builder(model)
        .tool(failing)
        .fail_on_tool_error(true)
        .build()
        .unwrap();

    let result = agent.run("try").await;
    assert!(matches!(
        result,
        Err(Error::Tool { tool, .. }) if tool == "explode"
    ));
}

#[tokio::test]
async fn tool_timeouts_are_recoverable() {
    let slow = function_tool(
        ToolSpec::new("slow", "Wait too long", object_schema()),
        |_arguments, _context| async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(ToolOutput::new("late"))
        },
    );
    let model = Arc::new(ScriptedModel::new([
        tool_response(vec![call("call_1", "slow", json!({}))]),
        text_response("recovered from timeout"),
    ]));
    let agent = Agent::builder(model.clone())
        .tool(slow)
        .tool_timeout_ms(Some(5))
        .build()
        .unwrap();

    let output = agent.run("try").await.unwrap();
    assert_eq!(output.text, "recovered from timeout");
    let requests = model.requests();
    let tool_result = requests[1].messages.last().unwrap();
    assert!(tool_result.text().contains("timed out after 5 ms"));
}

#[tokio::test]
async fn fail_on_tool_error_includes_timeouts() {
    let slow = function_tool(
        ToolSpec::new("slow", "Wait too long", object_schema()),
        |_arguments, _context| async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(ToolOutput::new("late"))
        },
    );
    let model = Arc::new(ScriptedModel::new([tool_response(vec![call(
        "call_1",
        "slow",
        json!({}),
    )])]));
    let agent = Agent::builder(model)
        .tool(slow)
        .tool_timeout_ms(Some(5))
        .fail_on_tool_error(true)
        .build()
        .unwrap();

    assert!(matches!(
        agent.run("try").await,
        Err(Error::ToolTimeout { tool, timeout_ms })
            if tool == "slow" && timeout_ms == 5
    ));
}

#[tokio::test]
async fn truncates_large_tool_outputs() {
    let long = function_tool(
        ToolSpec::new("long", "Return a long result", object_schema()),
        |_arguments, _context| async move { Ok(ToolOutput::new("abcdefghijklmnopqrstuvwxyz")) },
    );
    let model = Arc::new(ScriptedModel::new([
        tool_response(vec![call("call_1", "long", json!({}))]),
        text_response("truncated"),
    ]));
    let agent = Agent::builder(model.clone())
        .tool(long)
        .max_tool_output_chars(Some(10))
        .build()
        .unwrap();

    agent.run("try").await.unwrap();
    let result = model.requests()[1].messages.last().unwrap().text();
    assert!(result.starts_with("abcdefghij"));
    assert!(result.contains("truncated 16 chars"));
    assert!(!result.contains("uvwxyz"));
}

#[tokio::test]
async fn enforces_turn_limit() {
    let model = Arc::new(ScriptedModel::new([tool_response(vec![call(
        "call_1",
        "echo",
        json!({ "text": "loop" }),
    )])]));
    let agent = Agent::builder(model)
        .tool(echo_tool())
        .max_turns(1)
        .build()
        .unwrap();

    assert!(matches!(agent.run("loop").await, Err(Error::TurnLimit(1))));
}

#[tokio::test]
async fn cooperatively_cancels_running_tools() {
    let slow = function_tool(
        ToolSpec::new("slow", "Wait for cancellation", object_schema()),
        |_arguments, context| async move {
            context.cancellation().cancelled().await;
            Err(Error::Cancelled)
        },
    );
    let model = Arc::new(ScriptedModel::new([tool_response(vec![call(
        "call_1",
        "slow",
        json!({}),
    )])]));
    let agent = Arc::new(Agent::builder(model).tool(slow).build().unwrap());
    let cancellation = CancellationToken::new();
    let run = {
        let agent = agent.clone();
        let token = cancellation.clone();
        tokio::spawn(async move {
            agent
                .run_with("wait", RunOptions::default().with_cancellation(token))
                .await
        })
    };

    tokio::time::sleep(Duration::from_millis(20)).await;
    cancellation.cancel();
    let result = run.await.unwrap();
    assert!(matches!(result, Err(Error::Cancelled)));
}

#[tokio::test]
async fn cancels_in_flight_model_calls() {
    let model = BlockingModel::new();
    let started = model.started();
    let agent = Arc::new(Agent::builder(model).build().unwrap());
    let cancellation = CancellationToken::new();
    let run = {
        let agent = agent.clone();
        let token = cancellation.clone();
        tokio::spawn(async move {
            agent
                .run_with("wait", RunOptions::default().with_cancellation(token))
                .await
        })
    };

    started.notified().await;
    cancellation.cancel();
    let result = run.await.unwrap();
    assert!(matches!(result, Err(Error::Cancelled)));
}

#[tokio::test]
async fn cancels_run_while_waiting_for_the_agent_lock() {
    let model = BlockingModel::new();
    let started = model.started();
    let agent = Arc::new(Agent::builder(model).build().unwrap());
    let first_cancellation = CancellationToken::new();
    let second_cancellation = CancellationToken::new();

    let first = {
        let agent = agent.clone();
        let token = first_cancellation.clone();
        tokio::spawn(async move {
            agent
                .run_with("first", RunOptions::default().with_cancellation(token))
                .await
        })
    };
    started.notified().await;

    let second = {
        let agent = agent.clone();
        let token = second_cancellation.clone();
        tokio::spawn(async move {
            agent
                .run_with("second", RunOptions::default().with_cancellation(token))
                .await
        })
    };
    tokio::time::sleep(Duration::from_millis(10)).await;
    second_cancellation.cancel();
    assert!(matches!(second.await.unwrap(), Err(Error::Cancelled)));

    first_cancellation.cancel();
    assert!(matches!(first.await.unwrap(), Err(Error::Cancelled)));
}

#[tokio::test]
async fn persists_and_reloads_sessions() {
    let store = Arc::new(InMemorySessionStore::new());
    let first_model = Arc::new(ScriptedModel::new([text_response("first")]));
    let first = Agent::builder(first_model)
        .session_store(store.clone())
        .session_id("session-1")
        .build()
        .unwrap();
    first.run("remember this").await.unwrap();
    let saved = store.require("session-1").await.unwrap();
    assert_eq!(saved.messages.len(), 2);

    let second_model = Arc::new(ScriptedModel::new([text_response("second")]));
    let second = Agent::builder(second_model.clone())
        .session_store(store)
        .session_id("session-1")
        .build()
        .unwrap();
    second.run("what next").await.unwrap();

    let request = &second_model.requests()[0];
    let texts = request
        .messages
        .iter()
        .map(kestrel::Message::text)
        .collect::<Vec<_>>();
    assert!(texts.contains(&"remember this".to_owned()));
    assert!(texts.contains(&"first".to_owned()));
    assert!(texts.contains(&"what next".to_owned()));
}

#[tokio::test]
async fn deletes_persisted_sessions_without_stale_reload() {
    let store = Arc::new(InMemorySessionStore::new());
    let first_model = Arc::new(ScriptedModel::new([text_response("before delete")]));
    let agent = Agent::builder(first_model)
        .session_store(store.clone())
        .session_id("session-delete")
        .build()
        .unwrap();
    agent.run("remember this").await.unwrap();
    assert!(store.require("session-delete").await.is_ok());

    agent.delete_session().await.unwrap();
    assert!(store.load("session-delete").await.unwrap().is_none());
    assert!(agent.messages().await.is_empty());

    let second_model = Arc::new(ScriptedModel::new([text_response("after delete")]));
    let second = Agent::builder(second_model.clone())
        .session_store(store)
        .session_id("session-delete")
        .build()
        .unwrap();
    second.run("fresh start").await.unwrap();
    let texts = second_model.requests()[0]
        .messages
        .iter()
        .map(kestrel::Message::text)
        .collect::<Vec<_>>();
    assert!(texts.contains(&"fresh start".to_owned()));
    assert!(!texts.contains(&"remember this".to_owned()));
    assert!(!texts.contains(&"before delete".to_owned()));
}

#[tokio::test]
async fn resumes_an_interrupted_model_call_from_the_store() {
    let store = Arc::new(InMemorySessionStore::new());
    let blocking = BlockingModel::new();
    let started = blocking.started();
    let first = Agent::builder(blocking)
        .session_store(store.clone())
        .session_id("resume-model")
        .build()
        .unwrap();
    let cancellation = CancellationToken::new();
    let run = {
        let agent = first.clone();
        let token = cancellation.clone();
        tokio::spawn(async move {
            agent
                .run_with(
                    "finish this",
                    RunOptions::default().with_cancellation(token),
                )
                .await
        })
    };
    started.notified().await;
    cancellation.cancel();
    assert!(matches!(run.await.unwrap(), Err(Error::Cancelled)));

    let saved = store.require("resume-model").await.unwrap();
    let checkpoint = saved.active_run.expect("active run checkpoint");
    assert!(matches!(&checkpoint.phase, kestrel::RunPhase::ModelCall));
    assert_eq!(saved.messages.len(), 1);

    let model = Arc::new(ScriptedModel::new([text_response("resumed")]));
    let second = Agent::builder(model.clone())
        .session_store(store.clone())
        .session_id("resume-model")
        .build()
        .unwrap();
    let output = second.resume().await.unwrap();
    assert_eq!(output.run_id, checkpoint.run_id);
    assert_eq!(output.text, "resumed");
    assert_eq!(output.turns, 1);
    assert_eq!(model.requests().len(), 1);

    let saved = store.require("resume-model").await.unwrap();
    assert!(saved.active_run.is_none());
    assert_eq!(saved.messages.len(), 2);
    assert!(saved.revision > 0);
}

#[tokio::test]
async fn resumes_from_a_safe_tool_checkpoint() {
    let store = Arc::new(InMemorySessionStore::new());
    let mut session = Session::with_id("resume-tools");
    session.push(Message::user("use the tool"));
    let assistant = Message::assistant_with_tool_calls(
        "",
        vec![call("call-1", "echo", json!({ "text": "hello" }))],
    );
    session.active_run = Some(RunCheckpoint::tools(
        "run-tools",
        1,
        Usage::default(),
        assistant,
    ));
    session.touch();
    store.save(&session).await.unwrap();

    let model = Arc::new(ScriptedModel::new([text_response("tool complete")]));
    let agent = Agent::builder(model.clone())
        .tool(echo_tool())
        .session_store(store.clone())
        .session_id("resume-tools")
        .build()
        .unwrap();
    assert!(matches!(
        agent.run("new prompt").await,
        Err(Error::ActiveRun(run_id)) if run_id == "run-tools"
    ));

    let output = agent.resume().await.unwrap();
    assert_eq!(output.text, "tool complete");
    assert_eq!(output.turns, 2);
    assert_eq!(model.requests()[0].messages.last().unwrap().text(), "hello");

    let saved = store.require("resume-tools").await.unwrap();
    assert!(saved.active_run.is_none());
    assert_eq!(saved.messages.len(), 4);
}

#[tokio::test]
async fn refuses_to_replay_an_uncertain_tool_without_opt_in() {
    let store = Arc::new(InMemorySessionStore::new());
    let mut session = Session::with_id("resume-unsafe-tool");
    session.push(Message::user("use the tool"));
    let assistant = Message::assistant_with_tool_calls(
        "",
        vec![call("call-1", "echo", json!({ "text": "hello" }))],
    );
    let mut checkpoint = RunCheckpoint::tools("run-unsafe", 1, Usage::default(), assistant);
    checkpoint.mark_tool_started();
    session.active_run = Some(checkpoint);
    session.touch();
    store.save(&session).await.unwrap();

    let model = Arc::new(ScriptedModel::new([text_response("retried")]));
    let agent = Agent::builder(model.clone())
        .tool(echo_tool())
        .session_store(store)
        .session_id("resume-unsafe-tool")
        .build()
        .unwrap();

    assert!(matches!(
        agent.resume().await,
        Err(Error::UnsafeResume { tool, call_id })
            if tool == "echo" && call_id == "call-1"
    ));
    assert!(model.requests().is_empty());

    let output = agent
        .resume_with(ResumeOptions::default().with_uncertain_tool_retry(true))
        .await
        .unwrap();
    assert_eq!(output.text, "retried");
}

#[tokio::test]
async fn resume_requires_an_active_run() {
    let model = ScriptedModel::new([text_response("unused")]);
    let agent = Agent::builder(model)
        .session_id("no-active-run")
        .build()
        .unwrap();
    assert!(matches!(agent.resume().await, Err(Error::NoActiveRun)));
}

#[tokio::test]
async fn rejects_unsupported_persisted_session_schema() {
    let store = Arc::new(InMemorySessionStore::new());
    let mut session = Session::with_id("future-schema");
    session.schema_version = SESSION_SCHEMA_VERSION + 1;
    store.save(&session).await.unwrap();

    let model = ScriptedModel::new([text_response("unused")]);
    let agent = Agent::builder(model)
        .session_store(store)
        .session_id("future-schema")
        .build()
        .unwrap();
    assert!(matches!(
        agent.run("hello").await,
        Err(Error::UnsupportedSessionVersion { found, supported })
            if found == SESSION_SCHEMA_VERSION + 1 && supported == SESSION_SCHEMA_VERSION
    ));
}

#[tokio::test]
async fn reset_preserves_monotonic_session_revision() {
    let model = ScriptedModel::new([text_response("first")]);
    let agent = Agent::builder(model)
        .session_id("reset-revision")
        .build()
        .unwrap();
    agent.run("hello").await.unwrap();
    let before = agent.session().await.revision;
    assert!(before > 0);

    agent.reset().await.unwrap();
    let reset = agent.session().await;
    assert!(reset.revision > before);
    assert!(reset.messages.is_empty());
    assert!(reset.active_run.is_none());
}
