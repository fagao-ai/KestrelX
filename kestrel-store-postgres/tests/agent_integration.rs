use std::sync::Arc;

use async_trait::async_trait;
use kestrel::{
    Agent, FinishReason, Message, Model, ModelRequest, ModelResponse, Result, RunCheckpoint,
    Session, SessionStore, Usage,
};
use kestrel_store_postgres::PostgresSessionStore;
use uuid::Uuid;

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

fn test_url() -> Option<String> {
    std::env::var("KESTREL_POSTGRES_TEST_URL").ok()
}

#[tokio::test]
async fn works_as_an_agent_session_store() {
    let Some(url) = test_url() else {
        return;
    };
    let store = Arc::new(PostgresSessionStore::connect(&url).await.unwrap());
    let id = format!("kestrel-agent-{}", Uuid::new_v4());
    let agent = Agent::builder(StaticModel)
        .session_store(store.clone())
        .session_id(&id)
        .streaming(false)
        .build()
        .unwrap();

    let output = agent.run("hello").await.unwrap();
    assert_eq!(output.text, "stored answer");
    let session = store.load(&id).await.unwrap().unwrap();
    assert_eq!(session.messages.len(), 2);
    assert!(session.active_run.is_none());
    store.delete(&id).await.unwrap();
}

#[tokio::test]
async fn stores_and_resumes_an_interrupted_run() {
    let Some(url) = test_url() else {
        return;
    };
    let store = Arc::new(PostgresSessionStore::connect(&url).await.unwrap());
    let id = format!("kestrel-resume-{}", Uuid::new_v4());
    let run_id = format!("run-{}", Uuid::new_v4());
    let mut session = Session::with_id(&id);
    session.push(Message::user("finish interrupted work"));
    session.active_run = Some(RunCheckpoint::model_call(&run_id, 1, Usage::default()));
    session.touch();
    store.save(&session).await.unwrap();

    let agent = Agent::builder(StaticModel)
        .session_store(store.clone())
        .session_id(&id)
        .streaming(false)
        .build()
        .unwrap();
    let output = agent.resume().await.unwrap();
    assert_eq!(output.run_id, run_id);
    assert_eq!(output.text, "stored answer");
    assert!(store.load(&id).await.unwrap().unwrap().active_run.is_none());
    store.delete(&id).await.unwrap();
}
