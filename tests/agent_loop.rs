mod support;

use std::{sync::Arc, time::Duration};

use kestrel::{
    Agent, AgentEventKind, Error, InMemorySessionStore, RunOptions, ToolOutput, ToolSpec,
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
