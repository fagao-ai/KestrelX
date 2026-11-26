use std::sync::Arc;

use async_trait::async_trait;
use kestrel::{
    Agent, FinishReason, Message, Model, ModelRequest, ModelResponse, Result, RunCheckpoint,
    Session, SessionStore, Usage,
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

#[tokio::test]
async fn stores_and_resumes_an_interrupted_run() {
    let store = Arc::new(SqliteSessionStore::open_in_memory().unwrap());
    let mut session = Session::with_id("sqlite-resume");
    session.push(Message::user("finish interrupted work"));
    session.active_run = Some(RunCheckpoint::model_call(
        "sqlite-resume-run",
        1,
        Usage::default(),
    ));
    session.touch();
    store.save(&session).await.unwrap();

    let agent = Agent::builder(StaticModel)
        .session_store(store.clone())
        .session_id("sqlite-resume")
        .streaming(false)
        .build()
        .unwrap();
    let output = agent.resume().await.unwrap();
    assert_eq!(output.run_id, "sqlite-resume-run");
    assert_eq!(output.text, "stored answer");

    let session = store.load("sqlite-resume").await.unwrap().unwrap();
    assert!(session.active_run.is_none());
    assert_eq!(session.messages.len(), 2);
}
