use std::{io::Write, sync::Arc};

use kestrel::{
    Agent, AgentEventKind, InMemoryMemory, ModelDelta, OpenAiCompatible, SlidingWindowCompressor,
    TodoList, ToolOutput, ToolSpec, event_channel, function_tool,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_name = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4.1-mini".to_owned());
    let model = OpenAiCompatible::from_env(model_name)?;

    let lookup_order = function_tool(
        ToolSpec::new(
            "lookup_order",
            "Look up the current status of a customer order.",
            json!({
                "type": "object",
                "properties": {
                    "order_id": { "type": "string" }
                },
                "required": ["order_id"],
                "additionalProperties": false
            }),
        ),
        |arguments, _context| async move {
            let order_id = arguments["order_id"].as_str().unwrap_or("unknown");
            Ok(ToolOutput::new(format!(
                "Order {order_id}: packed and scheduled to ship tomorrow."
            )))
        },
    );

    let memory = Arc::new(InMemoryMemory::new());
    let todos = TodoList::new();
    let (events, mut event_stream) = event_channel();
    let agent = todos
        .install(Agent::builder(model))
        .system_prompt(
            "You are a customer support assistant. Be concise. Use tools for \
             order facts and preserve durable customer preferences.",
        )
        .tool(lookup_order)
        .memory(memory)
        .with_memory_tools()
        .compressor(Arc::new(SlidingWindowCompressor::new(80_000)))
        .event_sink(Arc::new(events))
        .max_turns(8)
        .build()?;

    let printer = tokio::spawn(async move {
        while let Some(event) = event_stream.recv().await {
            match event.kind {
                AgentEventKind::ModelDelta {
                    delta: ModelDelta::Text { text },
                    ..
                } => {
                    print!("{text}");
                    let _ = std::io::stdout().flush();
                }
                AgentEventKind::RunCompleted { .. } | AgentEventKind::RunFailed { .. } => {
                    println!();
                    break;
                }
                _ => {}
            }
        }
    });

    let output = agent
        .run("Where is order A-100? Track the investigation in a todo list.")
        .await?;
    printer.await?;
    eprintln!("turns={}, tokens={}", output.turns, output.usage.total());
    Ok(())
}
