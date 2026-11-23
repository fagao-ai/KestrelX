mod support;

use std::sync::Arc;

use kestrel::{
    Agent, AgentEventKind, CompressionRequest, ContextCompressor, HeuristicTokenCounter,
    InMemoryMemory, Memory, MemoryEntry, MemoryOptions, Message, Session, SlidingWindowCompressor,
    SummarizingCompressor,
};
use support::{ScriptedModel, text_response};

fn long_history() -> Session {
    let mut session = Session::new();
    for index in 0..12 {
        session.push(Message::user(format!(
            "Earlier user message {index} with enough text to consume context tokens."
        )));
        session.push(Message::assistant(format!(
            "Earlier assistant response {index} with enough text to consume context tokens."
        )));
    }
    session
}

#[tokio::test]
async fn compacts_context_before_model_call() {
    let model = Arc::new(ScriptedModel::new([text_response("compacted run")]));
    let (sender, mut receiver) = kestrel::event_channel();
    let agent = Agent::builder(model.clone())
        .session(long_history())
        .compressor(Arc::new(SlidingWindowCompressor::new(220)))
        .event_sink(Arc::new(sender))
        .build()
        .unwrap();

    agent.run("current question").await.unwrap();
    let request = &model.requests()[0];
    assert!(request.messages.iter().any(|message| {
        message.role == kestrel::Role::System && message.text().contains("Context compacted")
    }));
    assert!(
        request
            .messages
            .iter()
            .any(|message| message.text() == "current question")
    );

    let session = agent.session().await;
    assert_eq!(session.messages[0].role, kestrel::Role::System);
    assert!(session.messages[0].text().contains("Context compacted"));

    let mut compressed = None;
    while let Ok(event) = receiver.try_recv() {
        if let AgentEventKind::ContextCompressed {
            before_tokens,
            after_tokens,
            ..
        } = event.kind
        {
            compressed = Some((before_tokens, after_tokens));
        }
    }
    let (before, after) = compressed.expect("compression event");
    assert!(after <= 220);
    assert!(before > after);
}

#[tokio::test]
async fn summarizes_old_history_with_a_model() {
    let summarizer = Arc::new(ScriptedModel::new([text_response(
        "The user has been discussing context management.",
    )]));
    let model = Arc::new(ScriptedModel::new([text_response("answer")]));
    let compressor = SummarizingCompressor::new(summarizer.clone(), 300).with_summary_tokens(64);
    let agent = Agent::builder(model.clone())
        .session(long_history())
        .compressor(Arc::new(compressor))
        .build()
        .unwrap();

    agent.run("next").await.unwrap();
    assert_eq!(summarizer.requests().len(), 1);
    let request = &model.requests()[0];
    assert!(request.messages.iter().any(|message| {
        message.role == kestrel::Role::System
            && message.text().contains("Conversation summary")
            && message.text().contains("context management")
    }));
}

#[tokio::test]
async fn falls_back_to_sliding_compression_when_summary_fails() {
    let summarizer = Arc::new(ScriptedModel::new([]));
    let model = Arc::new(ScriptedModel::new([text_response("answer")]));
    let compressor = SummarizingCompressor::new(summarizer, 300).with_summary_tokens(64);
    let agent = Agent::builder(model.clone())
        .session(long_history())
        .compressor(Arc::new(compressor))
        .build()
        .unwrap();

    agent.run("next").await.unwrap();
    let request = &model.requests()[0];
    assert!(request.messages.iter().any(|message| {
        message.role == kestrel::Role::System && message.text().contains("Context compacted")
    }));
}

#[tokio::test]
async fn injects_relevant_long_term_memory() {
    let memory = Arc::new(InMemoryMemory::new());
    memory
        .remember(MemoryEntry::new("The project codename is Kestrel"))
        .await
        .unwrap();
    memory
        .remember(MemoryEntry::new("Unrelated lunch preference"))
        .await
        .unwrap();
    let model = Arc::new(ScriptedModel::new([text_response("ok")]));
    let (sender, mut receiver) = kestrel::event_channel();
    let agent = Agent::builder(model.clone())
        .memory(memory)
        .memory_options(MemoryOptions {
            auto_recall: true,
            recall_limit: 3,
            max_context_tokens: 200,
        })
        .event_sink(Arc::new(sender))
        .build()
        .unwrap();

    agent.run("What is the project codename?").await.unwrap();
    let request = &model.requests()[0];
    let memory_message = request
        .messages
        .iter()
        .find(|message| message.text().contains("<memory>"))
        .expect("memory message");
    assert!(memory_message.text().contains("Kestrel"));

    let mut recalled = false;
    while let Ok(event) = receiver.try_recv() {
        if matches!(event.kind, AgentEventKind::MemoryRecalled { count: 1 }) {
            recalled = true;
        }
    }
    assert!(recalled);
}

#[tokio::test]
async fn rejects_orphan_tool_results_in_loaded_history() {
    let model = Arc::new(ScriptedModel::new([]));
    let mut session = Session::new();
    session.push(Message::tool_result("missing", "orphan"));
    let agent = Agent::builder(model.clone())
        .session(session)
        .build()
        .unwrap();

    let result = agent.run("continue").await;
    assert!(matches!(result, Err(kestrel::Error::Context(_))));
    assert!(model.requests().is_empty());
}

#[tokio::test]
async fn never_splits_tool_calls_from_results() {
    let counter = HeuristicTokenCounter;
    let messages = vec![
        Message::user("one"),
        Message::assistant_with_tool_calls(
            "",
            vec![kestrel::ToolCall {
                id: "1".to_owned(),
                name: "lookup".to_owned(),
                arguments: serde_json::json!({}),
            }],
        ),
        Message::tool_result("1", "first result"),
        Message::user("two"),
        Message::assistant_with_tool_calls(
            "",
            vec![kestrel::ToolCall {
                id: "2".to_owned(),
                name: "lookup".to_owned(),
                arguments: serde_json::json!({}),
            }],
        ),
        Message::tool_result("2", "second result"),
        Message::user("latest"),
    ];
    let compressor = SlidingWindowCompressor::new(100);

    for budget in [8, 16, 24, 32, 48, 64] {
        let result = compressor
            .compress(CompressionRequest {
                messages: messages.clone(),
                budget_tokens: budget,
                token_counter: Arc::new(counter),
            })
            .await;
        let Ok(compression) = result else {
            continue;
        };
        for (index, message) in compression.messages.iter().enumerate() {
            if message.role != kestrel::Role::Tool {
                continue;
            }
            if index == 0 {
                panic!("orphan tool result at budget {budget}");
            }
            let parent = &compression.messages[index - 1];
            assert!(
                parent.role == kestrel::Role::Assistant || parent.role == kestrel::Role::Tool,
                "orphan tool result at budget {budget}"
            );
        }
    }
}
