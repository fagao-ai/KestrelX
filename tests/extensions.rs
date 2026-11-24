mod support;

use std::sync::Arc;

#[cfg(feature = "todo")]
use kestrel::TodoList;
use kestrel::{Agent, SubAgentTool};
use serde_json::json;
use support::{ScriptedModel, call, text_response, tool_response};

#[tokio::test]
async fn subagent_runs_with_isolated_conversation() {
    let model = Arc::new(ScriptedModel::new([
        tool_response(vec![call(
            "delegate-1",
            "delegate",
            json!({ "task": "Inspect the request" }),
        )]),
        text_response("sub agent answer"),
        text_response("parent final"),
    ]));
    let agent = Agent::builder(model.clone())
        .system_prompt("be useful")
        .build()
        .unwrap();
    let tool = SubAgentTool::from_agent(
        "delegate",
        "Delegate a self-contained task to a child agent.",
        &agent,
    );
    agent.register_tool(Arc::new(tool)).unwrap();

    let output = agent.run("parent task").await.unwrap();
    assert_eq!(output.text, "parent final");
    assert_eq!(model.requests().len(), 3);

    let child_request = &model.requests()[1];
    let child_text = child_request
        .messages
        .iter()
        .map(kestrel::Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(child_text.contains("Inspect the request"));
    assert!(!child_text.contains("parent task"));

    let parent_messages = agent.messages().await;
    let result = parent_messages
        .iter()
        .find(|message| message.role == kestrel::Role::Tool)
        .unwrap();
    assert_eq!(result.text(), "sub agent answer");
}

#[tokio::test]
#[cfg(feature = "todo")]
async fn todo_is_opt_in_but_fully_usable_when_installed() {
    let todos = TodoList::new();
    let model = Arc::new(ScriptedModel::new([
        tool_response(vec![call(
            "todo-1",
            "todo_write",
            json!({
                "todos": [
                    { "content": "Implement", "status": "completed" },
                    { "content": "Verify", "status": "in_progress" }
                ]
            }),
        )]),
        text_response("plan updated"),
    ]));
    let agent = Agent::builder(model).tools(todos.tools()).build().unwrap();

    let output = agent.run("make a plan").await.unwrap();
    assert_eq!(output.text, "plan updated");
    assert_eq!(todos.len(), 2);
    assert!(
        todos
            .snapshot()
            .iter()
            .any(|todo| todo.content == "Verify" && todo.status == kestrel::TodoStatus::InProgress)
    );
}
