use std::sync::Arc;

use async_trait::async_trait;
use kestrel::{
    Agent, FinishReason, Message, Model, ModelRequest, ModelResponse, Result, SessionStore, Usage,
};
use kestrel_store_sqlite::SqliteSessionStore;

struct StaticModel;

#[async_trait]
impl Model for StaticModel {
    fn name(&self) -> &str {
        "static"
    }

    async fn complete(&self, _request: ModelRequest) -> Result<ModelResponse> {
        Ok(ModelResponse {
            message: Message::assistant("stored answer"),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                input_tokens: 3,
                output_tokens: 2,
            },
            model: Some("static".to_owned()),
            raw: None,
        })
    }
}

#[tokio::test]
async fn works_as_an_agent_session_store() {
    let store = Arc::new(SqliteSessionStore::open_in_memory().unwrap());
    let agent = Agent::builder(StaticModel)
        .session_store(store.clone())
        .session_id("sqlite-agent")
        .streaming(false)
        .build()
        .unwrap();

    let output = agent.run("hello").await.unwrap();
    assert_eq!(output.text, "stored answer");

    let session = store.load("sqlite-agent").await.unwrap().unwrap();
    assert_eq!(session.messages.len(), 2);
    assert!(session.active_run.is_none());
    assert!(session.revision > 0);
}
